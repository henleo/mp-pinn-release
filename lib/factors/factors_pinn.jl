"""
EP factors for a single-hidden-unit nonlinear PINN `u = g(w·φ)`: the PDE-residual factor `f_R`
(`ResidualFactor`) and the data factor `f_U` (`DataObsFactor`), as in the paper's approach section.
Both share one contract -- `forward = X_forward_message(f); X_backward_update!(f, forward)` --
FORWARD moment-matches `w`'s current beliefs through the factor's own nonlinear constraint,
BACKWARD solves each `wₖ` from that shared result and installs its message.
"""

"""
    ResidualFactor

Residual `R = g'(z)·(w·r) - ρ·g(z) + ρ·g(z)²` at one collocation point (`z=w·φ`), `R ~ N(0,β²)`.
`ρ=0` recovers advection/heat's `g'(z)·(w·r)` exactly; `ρ≠0` adds Fisher-KPP's reaction term.
"""
struct ResidualFactor <: Factor
    db_m::DistributionBag{Gaussian1D}
    db_g::DistributionBag{Gaussian1D}
    w_idx::Vector{Int64}
    φ::Vector{Float64}
    r::Vector{Float64}
    β::Float64
    α::Float64
    ρ::Float64
    msg_to_w::Vector{Int64}
end
ResidualFactor(db_m, db_g, w_idx, φ, r, β, α; ρ = 0.0) =
    ResidualFactor(db_m, db_g, w_idx, φ, r, β, α, ρ, [add!(db_g) for _ in w_idx])

"""
    residual_forward_message(f::ResidualFactor) -> NamedTuple

FORWARD: `w`'s current beliefs propagated through the PDE operator and moment-matched into the
residual `R`'s mean/variance, plus every intermediate the backward step reuses. Exact via
Stein's lemma (`Cov(h(z),z) = σz2·E[h'(z)]`), except `Var_C` (`ρ≠0` only), which approximates
`Var(g(z)²)` by moment-matching `g(z)` to a Gaussian first.
"""
function residual_forward_message(f::ResidualFactor)
    K_pt = length(f.w_idx)
    μs = Vector{Float64}(undef, K_pt)
    σ2s = Vector{Float64}(undef, K_pt)
    μz, σz2 = 0.0, 0.0
    q0, Var_Q = 0.0, 0.0
    @inbounds for k in 1:K_pt
        marginal_k = f.db_m[f.w_idx[k]]
        μs[k] = Gaussian.mean(marginal_k)
        σ2s[k] = Gaussian.variance(marginal_k)
        μz += f.φ[k] * μs[k]
        σz2 += f.φ[k]^2 * σ2s[k]
        q0 += f.r[k] * μs[k]
        Var_Q += f.r[k]^2 * σ2s[k]
    end
    σz = sqrt(σz2)
    E_gp, E_gpp = piecewise_activation_derivative_moments(μz, σz, f.α)

    correction, Cov_QZ = 0.0, 0.0
    @inbounds for k in 1:K_pt
        σk = sqrt(σ2s[k])
        mean_dk, _ = piecewise_derivative_component_moments(μs[k], σk, f.φ[k], μz, σz, f.α)
        correction += (mean_dk - E_gp * μs[k]) * f.r[k]
        Cov_QZ += f.r[k] * f.φ[k] * σ2s[k]
    end
    # ρ≠0 (Fisher-KPP): reaction term -ρg(z)+ρg(z)², z-sensitivity -ρE[g'(z)]+2ρE[g(z)g'(z)].
    extra_var_bc = 0.0
    if f.ρ != 0.0
        E_g, Var_g = piecewise_activation_moments(μz, σz, f.α)
        E_g2 = Var_g + E_g^2
        E_g_gp = piecewise_activation_times_derivative_moment(μz, σz, f.α)
        correction += f.ρ * (E_g2 - E_g)
        H_Z = q0 * E_gpp - f.ρ * E_gp + 2f.ρ * E_g_gp

        # Var(Y^2) for Y ~ N(E_g, Var_g) is 4·E_g²·Var_g + 2·Var_g² (E[Y⁴]-E[Y²]² expanded).
        Var_C = 4 * E_g^2 * Var_g + 2 * Var_g^2
        extra_var_bc = f.ρ^2 * (Var_g - E_gp^2 * σz2) +
                       f.ρ^2 * (Var_C - (2E_g_gp)^2 * σz2)
    else
        H_Z = q0 * E_gpp
    end

    E_D = correction + E_gp * q0
    extra_var = H_Z^2 * σz2 + 2 * E_gp * H_Z * Cov_QZ + extra_var_bc
    Var_D = E_gp^2 * Var_Q + extra_var

    return (E_D = E_D, Var_D = Var_D, q0 = q0, Var_Q = Var_Q, σz2 = σz2, Cov_QZ = Cov_QZ,
            E_gp = E_gp, E_gpp = E_gpp, H_Z = H_Z, extra_var_bc = extra_var_bc,
            correction = correction, μs = μs, σ2s = σ2s)
end


"""
    residual_backward_message_k(f, forward, k) -> (μ, σ2)

Labels `wₖ`'s own linearization `R ≈ c_k·wₖ + rest` against `R ~ N(0, β²)` and solves for
`wₖ`'s message, `c_k := rₖ·E[g'(z)] + φₖ·H_Z`, `rest`'s moments the leave-one-out remainder of
`forward`'s shared sums.
"""
function residual_backward_message_k(f::ResidualFactor, forward, k)
    rk, φk = f.r[k], f.φ[k]
    μk, σk2 = forward.μs[k], forward.σ2s[k]

    c_k = rk * forward.E_gp + φk * forward.H_Z
    c_k = copysign(max(abs(c_k), 1e-3), c_k)

    q0_mk = forward.q0 - rk * μk
    VarQ_mk = forward.Var_Q - rk^2 * σk2
    σz2_mk = forward.σz2 - φk^2 * σk2
    CovQZ_mk = forward.Cov_QZ - rk * φk * σk2

    rest_mean = forward.E_gp * q0_mk - forward.H_Z * φk * μk + forward.correction
    rest_var = forward.E_gp^2 * VarQ_mk + forward.H_Z^2 * σz2_mk + 2 * forward.E_gp * forward.H_Z * CovQZ_mk +
               forward.extra_var_bc

    μ = -rest_mean / c_k
    σ2 = (f.β^2 + rest_var) / c_k^2
    return μ, σ2
end

"""
    residual_backward_update!(f, forward) -> max KL

Computes and installs every `wₖ`'s backward message into its marginal, one pass.
"""
function residual_backward_update!(f::ResidualFactor, forward)
    kl = 0.0
    @inbounds for k in eachindex(f.w_idx)
        μ_k, σ_k2 = residual_backward_message_k(f, forward, k)
        new_msg = (isfinite(μ_k) && isfinite(σ_k2) && σ_k2 > 0) ?
            Gaussian1DFromMeanVariance(μ_k, σ_k2) : Gaussian1DUniform()
        old_marginal = f.db_m[f.w_idx[k]]
        incoming_msg = old_marginal / f.db_g[f.msg_to_w[k]]
        new_marginal = incoming_msg * new_msg
        f.db_m[f.w_idx[k]] = new_marginal
        f.db_g[f.msg_to_w[k]] = new_msg
        kl = max(kl, Gaussian.KL_divergence(old_marginal, new_marginal))
    end
    return kl
end

"""
    DataObsFactor

Data factor `f_U` (the paper's data-graph subsection): observes `u_obs = g(w·φ) + ε` at one
labeled point (only the initial condition, in our PINN setting), `ε ~ N(0,β²)`.
"""
struct DataObsFactor <: Factor
    db_m::DistributionBag{Gaussian1D}
    db_g::DistributionBag{Gaussian1D}
    w_idx::Vector{Int64}
    φ::Vector{Float64}
    u_obs::Float64
    β::Float64
    α::Float64
    msg_to_w::Vector{Int64}
end
DataObsFactor(db_m, db_g, w_idx, φ, u_obs, β, α) =
    DataObsFactor(db_m, db_g, w_idx, φ, u_obs, β, α, [add!(db_g) for _ in w_idx])

"""
    data_forward_message(f::DataObsFactor) -> (E_z, Var_z, E_u, Var_u)

FORWARD: `z = w·φ`'s moments from `w`'s current incoming messages, then moment-matched through the
activation (`piecewise_activation_moments`) into the network's output belief `u = g(z)`. Mirrors
`residual_forward_message`.
"""
function data_forward_message(f::DataObsFactor)
    E_z, Var_z = 0.0, 0.0
    @inbounds for k in eachindex(f.w_idx)
        msg_from_w_k = f.db_m[f.w_idx[k]] / f.db_g[f.msg_to_w[k]]
        E_z += f.φ[k] * Gaussian.mean(msg_from_w_k)
        Var_z += f.φ[k]^2 * Gaussian.variance(msg_from_w_k)
    end
    E_u, Var_u = piecewise_activation_moments(E_z, sqrt(Var_z), f.α)
    return (E_z = E_z, Var_z = Var_z, E_u = E_u, Var_u = Var_u)
end

"""
    data_backward_message_k(f::DataObsFactor, forward, z_tgt, β_eff2, k) -> (μ, σ2)

BACKWARD: `wₖ` isolated from `z = w·φ ~ N(z_tgt, β_eff2)` -- `forward`'s shared sums with
`wₖ`'s own contribution subtracted back out (exact leave-one-out, `z = Σᵢφᵢwᵢ` a literal sum) --
where `z_tgt, β_eff2` are the observation labeled against `z` (`data_backward_update!`'s job,
computed once and shared across every `k`).
"""
function data_backward_message_k(f::DataObsFactor, forward, z_tgt, β_eff2, k)
    E_z, Var_z = forward.E_z, forward.Var_z
    φk = f.φ[k]
    msg_from_w_k = f.db_m[f.w_idx[k]] / f.db_g[f.msg_to_w[k]]
    E_z_mk = E_z - φk * Gaussian.mean(msg_from_w_k)
    Var_z_mk = Var_z - φk^2 * Gaussian.variance(msg_from_w_k)
    new_mean = (z_tgt - E_z_mk) / φk
    new_var = (β_eff2 + Var_z_mk) / φk^2
    return new_mean, new_var
end

"""
    data_backward_update!(f::DataObsFactor, forward) -> max KL

BACKWARD: labels `u_obs ~ N(u_obs, β²)` against `z`.
"""
function data_backward_update!(f::DataObsFactor, forward)
    slope, = piecewise_activation_derivative_moments(forward.E_z, sqrt(forward.Var_z), f.α)
    slope = copysign(max(abs(slope), 1e-3), slope)
    resid_var = forward.Var_u - slope^2 * forward.Var_z + f.β^2
    z_tgt = forward.E_z + (f.u_obs - forward.E_u) / slope
    β_eff2 = resid_var / slope^2

    kl = 0.0
    @inbounds for k in eachindex(f.w_idx)
        iszero(f.φ[k]) && continue
        old_marginal = f.db_m[f.w_idx[k]]
        new_mean, new_var = data_backward_message_k(f, forward, z_tgt, β_eff2, k)
        new_msg = (isfinite(new_mean) && isfinite(new_var) && new_var > 0) ?
            Gaussian1DFromMeanVariance(new_mean, new_var) : Gaussian1DUniform()
        msg_from_w_k = old_marginal / f.db_g[f.msg_to_w[k]]
        new_marginal = msg_from_w_k * new_msg
        f.db_m[f.w_idx[k]] = new_marginal
        f.db_g[f.msg_to_w[k]] = new_msg
        kl = max(kl, Gaussian.KL_divergence(old_marginal, new_marginal))
    end
    return kl
end
