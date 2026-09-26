"""
Wide (multi-neuron) versions of the fused EP factors in `factors_pinn.jl`, for the model

    u = Σ_{h=1..H} a_h · g(w_h · φ)

with `H` hidden units, FIXED readout weights `a` (distinct values break the permutation
symmetry between the neurons) and one eigenbasis weight vector `w_h` per neuron. Under the
fully factorized beliefs every neuron is independent of every other, so a wide factor simply
holds one existing single-neuron factor per neuron (reused as-is for the per-neuron forward
moments and for the message slots) and combines them:

  FORWARD   E[u] = Σ_h a_h E[g(z_h)],   Var[u] = Σ_h a_h² Var[g(z_h)]        (data factor)
            E[R] = Σ_h a_h E[D_h],      Var[R] = Σ_h a_h² Var[D_h]           (residual factor)
            with `(E[g(z_h)], Var[g(z_h)])` from `data_forward_message` and `(E[D_h], Var[D_h])`
            from `residual_forward_message`, exactly as in the single-neuron case.
  BACKWARD  neuron `h` sees the other neurons as a Gaussian "rest" (their forward moments,
            summed) and then takes EXACTLY the single-neuron backward step against the
            resulting effective observation:
              data:      u_h = (u_obs - Σ_{h'≠h} a_h' E[g(z_h')]) / a_h,
                         noise (β² + Σ_{h'≠h} a_h'² Var[g(z_h')]) / a_h²,
                         then `data_backward_update!`'s slope linearization + leave-one-out;
              residual:  c_k -> a_h c_k, rest mean -> a_h·rest_h + Σ_{h'≠h} a_h' E[D_h'],
                         rest var -> a_h²·rest_var_h + Σ_{h'≠h} a_h'² Var[D_h'],
                         then `residual_backward_message_k`'s μ = -rest/c_k, σ² = (β²+rest_var)/c_k².

With `H = 1`, `a = (1,)` every line reduces to the single-neuron factors bit for bit (the
combination terms are exact `+0.0` / `*1.0`); checked against a single-neuron run (same
epochs, KL history, eigen-posterior and predictive to 1e-12).

Same contract as the single-neuron factors: `forward = wide_X_forward_message(f);
wide_X_backward_update!(f, forward)`. The forward moments of ALL neurons are computed once from
the current beliefs, and every neuron's backward step uses that same snapshot (exactly as one
single-neuron factor computes its forward once and then updates all of its `K` weights from it).
"""

"""
    WideDataObsFactor

Data factor for `u_obs = Σ_h a_h g(w_h·φ) + ε`, `ε ~ N(0,β²)`, at one labeled point. `neurons[h]`
is an ordinary `DataObsFactor` over neuron `h`'s weights (same `φ`, `u_obs`, `β`, `α` for all).
"""
struct WideDataObsFactor <: Factor
    neurons::Vector{DataObsFactor}
    a::Vector{Float64}
    function WideDataObsFactor(neurons, a)
        length(neurons) == length(a) || throw(ArgumentError("one readout weight per neuron"))
        new(neurons, a)
    end
end

"""
    wide_data_forward_message(f::WideDataObsFactor) -> NamedTuple

FORWARD: `data_forward_message` per neuron (`neurons[h]`), combined into the output belief
`E_u = Σ_h a_h E[g(z_h)]`, `Var_u = Σ_h a_h² Var[g(z_h)]`.
"""
function wide_data_forward_message(f::WideDataObsFactor)
    fwds = [data_forward_message(n) for n in f.neurons]
    E_u, Var_u = 0.0, 0.0
    @inbounds for h in eachindex(fwds)
        E_u += f.a[h] * fwds[h].E_u
        Var_u += f.a[h]^2 * fwds[h].Var_u
    end
    return (neurons = fwds, E_u = E_u, Var_u = Var_u)
end

"""
    wide_data_backward_update!(f::WideDataObsFactor, forward) -> max KL

BACKWARD: for every neuron `h`, the other neurons' forward moments form a Gaussian rest, giving
neuron `h` the effective observation `u_h ~ N(u_h, β_h²)`; the rest of the step is
`data_backward_update!` line for line with `u_obs -> u_h`, `β² -> β_h²`.
"""
function wide_data_backward_update!(f::WideDataObsFactor, forward)
    kl = 0.0
    for h in eachindex(f.neurons)
        n = f.neurons[h]
        fwd = forward.neurons[h]
        a_h = f.a[h]
        rest_mean, rest_var = 0.0, 0.0
        @inbounds for h2 in eachindex(f.neurons)
            h2 == h && continue
            rest_mean += f.a[h2] * forward.neurons[h2].E_u
            rest_var += f.a[h2]^2 * forward.neurons[h2].Var_u
        end
        u_obs_h = (n.u_obs - rest_mean) / a_h
        β2_h = (n.β^2 + rest_var) / a_h^2

        # --- from here: data_backward_update! with u_obs -> u_obs_h and β² -> β2_h ---
        slope, = piecewise_activation_derivative_moments(fwd.E_z, sqrt(fwd.Var_z), n.α)
        slope = copysign(max(abs(slope), 1e-3), slope)
        resid_var = fwd.Var_u - slope^2 * fwd.Var_z + β2_h
        z_tgt = fwd.E_z + (u_obs_h - fwd.E_u) / slope
        β_eff2 = resid_var / slope^2

        @inbounds for k in eachindex(n.w_idx)
            iszero(n.φ[k]) && continue
            old_marginal = n.db_m[n.w_idx[k]]
            new_mean, new_var = data_backward_message_k(n, fwd, z_tgt, β_eff2, k)
            new_msg = (isfinite(new_mean) && isfinite(new_var) && new_var > 0) ?
                Gaussian1DFromMeanVariance(new_mean, new_var) : Gaussian1DUniform()
            msg_from_w_k = old_marginal / n.db_g[n.msg_to_w[k]]
            new_marginal = msg_from_w_k * new_msg
            n.db_m[n.w_idx[k]] = new_marginal
            n.db_g[n.msg_to_w[k]] = new_msg
            kl = max(kl, Gaussian.KL_divergence(old_marginal, new_marginal))
        end
    end
    return kl
end

"""
    WideResidualFactor

PDE-residual factor for `R = Σ_h a_h g'(z_h)·(w_h·r) − ρ·U·(1 − U) ~ N(0,β²)`,
`U = Σ_h a_h g(z_h)`, at one collocation point. `ρ = 0` is advection/heat; `ρ ≠ 0` is the
Fisher-KPP reaction term, the wide version of `ResidualFactor`'s `ρ ≠ 0` path. Each
`neurons[h]` is an ordinary `ResidualFactor` built with `ρ = 0` (its forward moments are those
of the neuron's linear-operator part `A_h = g'(z_h)·(w_h·r)`); the reaction strength lives in
the wide factor (`ρ`), because the reaction couples the neurons through `U`.

Reaction term with a Gaussian leave-one-neuron-out rest (the same construction as the data and
`ρ = 0` residual factors, now applied to a quadratic coupling): for neuron `h` write
`U = a_h g_h + S`, `S = Σ_{h'≠h} a_h' g_h'` with mean `m_S` and variance `v_S` from the other
neurons' activation moments (independent under the factorized beliefs). Then

    U − U² = a_h g_h − a_h² g_h² − 2 a_h g_h S + (S − S²),

so neuron `h` sees its own residual `A_h − ρ₁ g_h + ρ₂ g_h² + 2ρ ξ g_h` with the effective
coefficients `ρ₁ = ρ (1 − 2 m_S)`, `ρ₂ = ρ a_h` and the zero-mean random coefficient
`ξ = S − m_S` (its product with `g_h` is uncorrelated with `z_h`, and adds the variance
`4ρ² v_S E[g_h²]`), while the other neurons contribute `L_S − ρ (S − S²)` (`L_S` = their
linear-operator parts) as a Gaussian rest with `S` moment-matched to a Gaussian:
`E[S − S²] = m_S − m_S² − v_S`, `Var[S − S²] = v_S (1 − 2 m_S)² + 2 v_S²`, and
`Cov(L_S, S − S²) = (1 − 2 m_S) Σ_{h'≠h} a_h'² Cov(A_h', g_h')` with the per-neuron covariance
from the same linearization in `z` the single-neuron factor uses
(`Cov(A, g) ≈ H_Z E[g'] σz² + E[g']² Cov(Q, z)`). With `(ρ₁, ρ₂)` in place of `(ρ, ρ)` the
single-neuron formulas apply unchanged (mean correction `−ρ₁E[g] + ρ₂E[g²]`, sensitivity
`H_Z = q₀E[g''] − ρ₁E[g'] + 2ρ₂E[g g']`, nonlinear remainder variances), so `H = 1, a = (1)`
reduces to the single-neuron `ρ ≠ 0` factor (checked to 1e-12 against the single-neuron
factor); for `ρ = 0` no reaction term enters and the factor is the plain-sum combination
above.
"""
struct WideResidualFactor <: Factor
    neurons::Vector{ResidualFactor}
    a::Vector{Float64}
    ρ::Float64
    function WideResidualFactor(neurons, a, ρ = 0.0)
        length(neurons) == length(a) || throw(ArgumentError("one readout weight per neuron"))
        all(n -> n.ρ == 0.0, neurons) || throw(ArgumentError("build the neurons with ρ = 0; the reaction strength is the wide factor's own ρ"))
        new(neurons, a, ρ)
    end
end

"""
    wide_residual_forward_message(f::WideResidualFactor) -> NamedTuple

FORWARD: `residual_forward_message` per neuron (the linear-operator parts `A_h`), combined into
`E_R = Σ_h a_h E[A_h] − ρ·E[U − U²]`, `Var_R = Σ_h a_h² Var[A_h] + ρ²·Var[U − U²]
− 2ρ·Cov(L, U − U²)` with `U` moment-matched to a Gaussian (`ρ = 0`: the plain sums). For
`ρ ≠ 0` the per-neuron activation moments (`E_g`, `Var_g`, `E_g2`, `E_g_gp`, `Cov_Ag`) the
backward step needs are computed here once.
"""
function wide_residual_forward_message(f::WideResidualFactor)
    fwds = [residual_forward_message(n) for n in f.neurons]
    E_R, Var_R = 0.0, 0.0
    @inbounds for h in eachindex(fwds)
        E_R += f.a[h] * fwds[h].E_D
        Var_R += f.a[h]^2 * fwds[h].Var_D
    end
    f.ρ == 0.0 && return (neurons = fwds, E_R = E_R, Var_R = Var_R, act = nothing)

    H = length(f.neurons)
    E_g = Vector{Float64}(undef, H); Var_g = similar(E_g); E_g2 = similar(E_g)
    E_g_gp = similar(E_g); Cov_Ag = similar(E_g)
    m_U, v_U, cov_LU = 0.0, 0.0, 0.0
    @inbounds for h in 1:H
        n, fwd = f.neurons[h], fwds[h]
        μz = 0.0
        for k in eachindex(n.w_idx)
            μz += n.φ[k] * fwd.μs[k]
        end
        σz = sqrt(fwd.σz2)
        E_g[h], Var_g[h] = piecewise_activation_moments(μz, σz, n.α)
        E_g2[h] = Var_g[h] + E_g[h]^2
        E_g_gp[h] = piecewise_activation_times_derivative_moment(μz, σz, n.α)
        Cov_Ag[h] = fwd.H_Z * fwd.E_gp * fwd.σz2 + fwd.E_gp^2 * fwd.Cov_QZ
        m_U += f.a[h] * E_g[h]
        v_U += f.a[h]^2 * Var_g[h]
        cov_LU += f.a[h]^2 * Cov_Ag[h]
    end
    E_R -= f.ρ * (m_U - m_U^2 - v_U)
    Var_R += f.ρ^2 * (v_U * (1 - 2m_U)^2 + 2v_U^2) - 2f.ρ * (1 - 2m_U) * cov_LU
    return (neurons = fwds, E_R = E_R, Var_R = Var_R,
            act = (E_g = E_g, Var_g = Var_g, E_g2 = E_g2, E_g_gp = E_g_gp, Cov_Ag = Cov_Ag))
end

"""
    wide_residual_backward_message_k(f, forward, h, k, rest_mean_others, rest_var_others) -> (μ, σ2)

`residual_backward_message_k` for neuron `h`'s coordinate `k`, with the readout weight folded into
the linearization coefficient (`c_k -> a_h c_k`) and the other neurons' forward moments
(`rest_mean_others = Σ_{h'≠h} a_h' E[D_h']`, `rest_var_others = Σ_{h'≠h} a_h'² Var[D_h']`) added
to the leave-one-out remainder. The 1e-3 floor is applied to the product `|a_h c_k|` (not to
`|c_k|` alone), which for `|a_h| = 1` is identical to the single-neuron clamp on `|c_k|`.
"""
function wide_residual_backward_message_k(f::WideResidualFactor, forward, h, k, rest_mean_others, rest_var_others;
                                          H_Z = forward.neurons[h].H_Z, correction = forward.neurons[h].correction,
                                          extra_var_bc = forward.neurons[h].extra_var_bc)
    n = f.neurons[h]
    fwd = forward.neurons[h]
    a_h = f.a[h]
    rk, φk = n.r[k], n.φ[k]
    μk, σk2 = fwd.μs[k], fwd.σ2s[k]

    c_k = a_h * (rk * fwd.E_gp + φk * H_Z)
    c_k = copysign(max(abs(c_k), 1e-3), c_k)

    q0_mk = fwd.q0 - rk * μk
    VarQ_mk = fwd.Var_Q - rk^2 * σk2
    σz2_mk = fwd.σz2 - φk^2 * σk2
    CovQZ_mk = fwd.Cov_QZ - rk * φk * σk2

    rest_mean_h = fwd.E_gp * q0_mk - H_Z * φk * μk + correction
    rest_var_h = fwd.E_gp^2 * VarQ_mk + H_Z^2 * σz2_mk + 2 * fwd.E_gp * H_Z * CovQZ_mk +
                 extra_var_bc

    rest_mean = a_h * rest_mean_h + rest_mean_others
    rest_var = a_h^2 * rest_var_h + rest_var_others

    μ = -rest_mean / c_k
    σ2 = (n.β^2 + rest_var) / c_k^2
    return μ, σ2
end

"""
    wide_reaction_terms(f, forward, h) -> (rest_mean_others, rest_var_others, H_Z, correction, extra_var_bc)

The `ρ ≠ 0` quantities of neuron `h`'s backward step (see `WideResidualFactor`): the other
neurons' rest `L_S − ρ(S − S²)` (mean, variance), and neuron `h`'s own sensitivity `H_Z`, mean
correction and nonlinear-remainder variance with the effective coefficients `ρ₁ = ρ(1 − 2m_S)`,
`ρ₂ = ρ a_h` plus the random-coefficient variance `4ρ² v_S E[g_h²]`.
"""
function wide_reaction_terms(f::WideResidualFactor, forward, h)
    act = forward.act
    ρ, a_h = f.ρ, f.a[h]
    fwd = forward.neurons[h]
    m_S, v_S, m_L, v_L, cov_LS = 0.0, 0.0, 0.0, 0.0, 0.0
    @inbounds for h2 in eachindex(f.neurons)
        h2 == h && continue
        m_S += f.a[h2] * act.E_g[h2]
        v_S += f.a[h2]^2 * act.Var_g[h2]
        m_L += f.a[h2] * forward.neurons[h2].E_D
        v_L += f.a[h2]^2 * forward.neurons[h2].Var_D
        cov_LS += f.a[h2]^2 * act.Cov_Ag[h2]
    end
    rest_mean_others = m_L - ρ * (m_S - m_S^2 - v_S)
    rest_var_others = max(v_L + ρ^2 * (v_S * (1 - 2m_S)^2 + 2v_S^2) - 2ρ * (1 - 2m_S) * cov_LS, 0.0)

    ρ1 = ρ * (1 - 2m_S)
    ρ2 = ρ * a_h
    E_g, Var_g, E_g2, E_g_gp = act.E_g[h], act.Var_g[h], act.E_g2[h], act.E_g_gp[h]
    correction = fwd.correction - ρ1 * E_g + ρ2 * E_g2
    H_Z = fwd.H_Z - ρ1 * fwd.E_gp + 2ρ2 * E_g_gp
    Var_C = 4 * E_g^2 * Var_g + 2 * Var_g^2           # Var(Y²) for Y ~ N(E_g, Var_g), as in the single-neuron factor
    extra_var_bc = ρ1^2 * (Var_g - fwd.E_gp^2 * fwd.σz2) + ρ2^2 * (Var_C - (2E_g_gp)^2 * fwd.σz2) +
                   4ρ^2 * v_S * E_g2
    return rest_mean_others, rest_var_others, H_Z, correction, extra_var_bc
end

"""
    wide_residual_backward_update!(f::WideResidualFactor, forward) -> max KL

Computes and installs every neuron's every `w_{h,k}` backward message, one pass
(`residual_backward_update!` per neuron with the other neurons' rest added).
"""
function wide_residual_backward_update!(f::WideResidualFactor, forward)
    kl = 0.0
    for h in eachindex(f.neurons)
        n = f.neurons[h]
        if f.ρ == 0.0
            rest_mean_others, rest_var_others = 0.0, 0.0
            @inbounds for h2 in eachindex(f.neurons)
                h2 == h && continue
                rest_mean_others += f.a[h2] * forward.neurons[h2].E_D
                rest_var_others += f.a[h2]^2 * forward.neurons[h2].Var_D
            end
            H_Z, correction, extra_var_bc = forward.neurons[h].H_Z, forward.neurons[h].correction, forward.neurons[h].extra_var_bc
        else
            rest_mean_others, rest_var_others, H_Z, correction, extra_var_bc = wide_reaction_terms(f, forward, h)
        end
        @inbounds for k in eachindex(n.w_idx)
            μ_k, σ_k2 = wide_residual_backward_message_k(f, forward, h, k, rest_mean_others, rest_var_others;
                                                         H_Z, correction, extra_var_bc)
            new_msg = (isfinite(μ_k) && isfinite(σ_k2) && σ_k2 > 0) ?
                Gaussian1DFromMeanVariance(μ_k, σ_k2) : Gaussian1DUniform()
            old_marginal = n.db_m[n.w_idx[k]]
            incoming_msg = old_marginal / n.db_g[n.msg_to_w[k]]
            new_marginal = incoming_msg * new_msg
            n.db_m[n.w_idx[k]] = new_marginal
            n.db_g[n.msg_to_w[k]] = new_msg
            kl = max(kl, Gaussian.KL_divergence(old_marginal, new_marginal))
        end
    end
    return kl
end
