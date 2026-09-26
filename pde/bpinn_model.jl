"""
Bayesian PINN baseline (`BPINN`): the SAME single-hidden-unit architecture `u = g(w·φ)` and 2D
Fourier basis `mp_pinn_model.jl`/`gd_pinn_model.jl` use, but with `w`'s posterior sampled by
Hamiltonian Monte Carlo as in Yang, Meng & Karniadakis 2021 (B-PINNs, J. Comput. Phys.), instead
of loopy EP (MP-PINN) or point-estimate Adam (GD-PINN). Reuses `GDPINN`'s activation functions and analytic-gradient
chain-rule structure -- the log-posterior gradient here differs from `GDPINN.pinn_loss_grad` only
by using the literal (sum-form) Gaussian log-likelihood instead of a mean-normalized loss, and by
adding the Gaussian-prior gradient term, so the target density matches MP-PINN's actual generative
model (`DataObsFactor`/`ResidualFactor`'s per-point noise `β_ic`/`β_pde`, and `build_graph`'s
zero-mean prior std `σw`) exactly. No autodiff/NN framework needed, consistent with the rest of
this codebase.
"""
module BPINN

include(joinpath(@__DIR__, "gd_pinn_model.jl"))

using .GDPINN
using .GDPINN.MPPINN
using .GDPINN.MPPINN.Factors
using LinearAlgebra
using Random
using Printf
using Optimisers

export neg_log_posterior_and_grad, find_map, predict_bpinn_stats, predict_residual_bpinn,
       leapfrog_mass, hmc_momentum, hmc_kinetic, hg_initial_step_size, DualAveraging, adapt_step!,
       adapted_step_size, refine_map_gauss_newton

"""
    neg_log_posterior_and_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; reaction = 0.0) -> (nlp, grad)

Literal (sum-form, not mean-normalized) Gaussian negative log posterior
`nlp = 0.5·Σ(resid_ic)²/β_ic² + 0.5·Σ(R_r)²/β_pde² + 0.5·‖w‖²/σw²` and its analytic gradient --
the same chain-rule/product-rule structure as `GDPINN.pinn_loss_grad`, but targeting the exact
probabilistic model `DataObsFactor`/`ResidualFactor`/`build_graph`'s prior encode (per-point noise
`β_ic`/`β_pde`, zero-mean Gaussian prior std `σw`), not a loss-balancing reweighting of it.
`reaction` = ρ of the residual factors (`GDPINN.reaction_strength(graph)`): 0 for advection;
for the Fisher-KPP graphs the residual is `g'(z)·Q − ρ·g(z)·(1−g(z))`, exactly the model form
`ResidualFactor` encodes (including its omitted g'' diffusion term).
"""
function neg_log_posterior_and_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; reaction = 0.0)
    z_ic = Φ_ic * w
    resid_ic = piecewise_activation.(z_ic, α) .- y_ic
    gp_ic = piecewise_activation_derivative.(z_ic, α)

    z_r = Φ_r * w
    Q_r = R_mat * w
    gp_r = piecewise_activation_derivative.(z_r, α)
    gpp_r = GDPINN.piecewise_activation_second_derivative.(z_r, α)
    R_r = gp_r .* Q_r

    grad_data = (Φ_ic' * (resid_ic .* gp_ic)) ./ β_ic^2
    grad_prior = w ./ σw^2
    if reaction == 0.0
        nlp = 0.5 * sum(abs2, resid_ic) / β_ic^2 + 0.5 * sum(abs2, R_r) / β_pde^2 +
              0.5 * sum(abs2, w) / σw^2
        grad_pde = (Φ_r' * (R_r .* gpp_r .* Q_r) .+ R_mat' * (R_r .* gp_r)) ./ β_pde^2
    else
        # Fisher-KPP model form of `ResidualFactor` (ρ ≠ 0): R = g'(z)·Q − ρ·g(z)·(1 − g(z)), so
        # ∂R/∂z at fixed Q gains −ρ·(1 − 2g)·g'.
        g_r = piecewise_activation.(z_r, α)
        R_r = R_r .- reaction .* g_r .* (1 .- g_r)
        dR_dz = gpp_r .* Q_r .- reaction .* (1 .- 2 .* g_r) .* gp_r
        nlp = 0.5 * sum(abs2, resid_ic) / β_ic^2 + 0.5 * sum(abs2, R_r) / β_pde^2 +
              0.5 * sum(abs2, w) / σw^2
        grad_pde = (Φ_r' * (R_r .* dR_dz) .+ R_mat' * (R_r .* gp_r)) ./ β_pde^2
    end
    return nlp, grad_data .+ grad_pde .+ grad_prior
end

"""
    find_map(w0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; max_iters, lr0, window, plateau_tol,
             decay_factor, max_decays, check_every) -> (w, iters, nlp_history)

MAP point of the exact posterior `neg_log_posterior_and_grad` targets, by Adam with the same
plateau/early-stopping schedule as `GDPINN.train_gd!`. Used to start the HMC chains in a
high-density region (standard practice for B-PINNs, and principled here since it's the MAP of the
same model HMC samples, not a different objective).
"""
function find_map(w0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α;
                   max_iters = 30_000, lr0 = 3e-4, window = 500, plateau_tol = 1e-3,
                   decay_factor = 0.5, max_decays = 4, check_every = 1000, reaction = 0.0)
    w = copy(w0)
    lr = lr0
    opt_state = Optimisers.setup(Optimisers.Adam(lr), w)
    nlp_history = Float64[]
    prev_window_mean = Inf
    decays = 0
    iters_done = 0
    for iter in 1:max_iters
        nlp, grad = neg_log_posterior_and_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; reaction)
        push!(nlp_history, nlp)
        opt_state, w = Optimisers.update!(opt_state, w, grad)
        iters_done = iter
        if iter % check_every == 0
            @printf("  [MAP] iter %6d  nlp %.6e  lr %.2e\n", iter, nlp, lr)
            flush(stdout)
        end
        if iter % window == 0
            window_mean = sum(@view nlp_history[(end - window + 1):end]) / window
            if window_mean > prev_window_mean * (1 - plateau_tol)
                decays >= max_decays && break
                decays += 1
                lr *= decay_factor
                Optimisers.adjust!(opt_state, lr)
            end
            prev_window_mean = window_mean
        end
    end
    return w, iters_done, nlp_history
end

"""
    accept_prob(H0, H_prop) -> Float64

Metropolis-Hastings acceptance probability `min(1, exp(H0 - H_prop))` of a proposal with
Hamiltonian `H_prop` from a state with Hamiltonian `H0` (the standard Metropolis rule, Neal 2011).
Clamped to exactly 0 (rather than NaN) whenever the proposal diverged (`H_prop` non-finite -- a
real risk during the initial step-size search, where large candidate step sizes can overflow the
piecewise activation). Yang et al. 2021 print the acceptance test of their Algorithm 1
with the energy difference and the comparison both reversed; the standard rule is used here.
"""
function accept_prob(H0, H_prop)
    isfinite(H_prop) || return 0.0
    return min(1.0, exp(min(0.0, H0 - H_prop)))
end

# ---------------------------------------------------------------------------------------------
# Dense-mass-matrix HMC helpers shared by the multichain runners
# (`pde/advection/bpinn_run_beta_multichain.jl`, `pde/diffusion/bpinn_run_reacdiff_multichain.jl`).
# `mass_chol` is the Cholesky
# factorization of the mass matrix M = L·Lᵀ (the Gauss-Newton curvature at the MAP in those
# runners): momentum p ~ N(0, M), kinetic energy ½ pᵀM⁻¹p, position step ε·M⁻¹p.

"Momentum draw p ~ N(0, M) for M = L·Lᵀ."
hmc_momentum(rng, mass_chol) = mass_chol.L * randn(rng, size(mass_chol, 1))

"Kinetic energy ½ pᵀ M⁻¹ p, via one triangular solve."
hmc_kinetic(pvec, mass_chol) = 0.5 * sum(abs2, mass_chol.L \ pvec)

"""
    leapfrog_mass(w, p, ε, L, grad_fn, mass_chol) -> (w, p)

Leapfrog integration of `L` steps of size `ε` with a dense mass matrix: one leading momentum
half-kick, `L` position steps `ε·M⁻¹p` (two triangular solves each) each followed by a full kick
except the last, which is followed by a half-kick.
"""
function leapfrog_mass(w, pvec, ε, L, grad_fn, mass_chol)
    w = copy(w); pvec = copy(pvec)
    _, grad = grad_fn(w)
    pvec .-= (ε / 2) .* grad
    for step in 1:L
        w .+= ε .* (mass_chol \ pvec)
        _, grad = grad_fn(w)
        pvec .-= (step == L ? ε / 2 : ε) .* grad
    end
    return w, pvec
end

"""
    hg_initial_step_size(w0, grad_fn, mass_chol; L, rng) -> Float64

Hoffman & Gelman 2014, Algorithm 4, under the dense-mass metric: ONE momentum draw, then ε is
doubled or halved from 1 until the acceptance probability of an `L`-step trajectory crosses 0.5.
Returns the LAST step size on the accepting side of that crossing in both directions: when
halving, the first ε with α ≥ 0.5; when doubling, the last ε with α > 0.5. The result is only
the starting value of the warm-up adaptation (`DualAveraging`).
"""
function hg_initial_step_size(w0, grad_fn, mass_chol; L, rng)
    nlp_w, _ = grad_fn(w0)
    p0 = hmc_momentum(rng, mass_chol)
    H0 = nlp_w + hmc_kinetic(p0, mass_chol)
    function α_at(ε)
        w_prop, p_prop = leapfrog_mass(w0, p0, ε, L, grad_fn, mass_chol)
        nlp_prop, _ = grad_fn(w_prop)
        return accept_prob(H0, nlp_prop + hmc_kinetic(p_prop, mass_chol))
    end
    ε = 1.0
    direction = α_at(ε) > 0.5 ? 1 : -1
    for _ in 1:100  # bounded so a pathological target can't loop forever; falls back to the last ε tried
        ε_next = ε * 2.0^direction
        α = α_at(ε_next)
        if direction == 1
            α <= 0.5 && return ε        # ε_next just failed: keep the last accepting step
        else
            α >= 0.5 && return ε_next   # ε_next is the first accepting step
        end
        ε = ε_next
    end
    return ε
end

"""
    DualAveraging(ε0; δ = 0.8, γ = 0.05, t0 = 10, κ = 0.75)

Step-size adaptation for the warm-up (burn-in) phase of a chain: the dual-averaging scheme of
Hoffman & Gelman 2014 (Section 3.2, Algorithm 5's ε-update, with their default constants),
driving the running mean of the Metropolis acceptance probability to the target `δ`. After each
warm-up proposal with acceptance probability `α`, `adapt_step!(da, α)` returns the step size for
the next proposal; once warm-up ends, `adapted_step_size(da)` is the averaged iterate
`exp(log ε̄)` that the chain keeps FIXED for all retained draws (so the retained draws come from
an ordinary fixed-step HMC transition kernel). `ε0` is the initial value (from
`hg_initial_step_size`); `μ = log(10 ε0)` biases the early iterates upward as in the paper.

Why adaptation and not a fixed step from a one-off search: the leapfrog integrator's acceptance
collapses beyond a stability limit that depends on the local curvature, and with the
Gauss-Newton metric evaluated at the MAP that limit differs between the mode and the bulk of
the posterior. A step size that passes a probe at the mode can freeze the chain a few
iterations later. Adapting on the acceptance actually observed along the chain during warm-up is the
standard remedy (Stan's warm-up does the same); the target 0.8 is Stan's default and is
conservative relative to the 0.65 optimum of Beskos et al. 2013.
"""
mutable struct DualAveraging
    μ::Float64
    γ::Float64
    t0::Float64
    κ::Float64
    δ::Float64
    t::Int
    H̄::Float64
    logε̄::Float64
    logε::Float64
end
DualAveraging(ε0; δ = 0.8, γ = 0.05, t0 = 10.0, κ = 0.75) =
    DualAveraging(log(10 * ε0), γ, t0, κ, δ, 0, 0.0, 0.0, log(ε0))

"One warm-up update with the acceptance probability `α` of the proposal just made; returns the next step size."
function adapt_step!(da::DualAveraging, α)
    da.t += 1
    η = 1 / (da.t + da.t0)
    da.H̄ = (1 - η) * da.H̄ + η * (da.δ - α)
    da.logε = da.μ - sqrt(da.t) / da.γ * da.H̄
    w = da.t^(-da.κ)
    da.logε̄ = w * da.logε + (1 - w) * da.logε̄
    return exp(da.logε)
end

"The averaged step size to keep fixed after warm-up."
adapted_step_size(da::DualAveraging) = exp(da.logε̄)

"""
    refine_map_gauss_newton(w, nlp_and_grad, gauss_newton; max_iters = 50, λ0 = 1e-3,
                            rel_tol = 1e-10, grad_tol = 1e-8) -> (w, iters, nlp_history)

Damped Gauss-Newton (Levenberg-Marquardt) polish of a MAP estimate: from the Adam warm start,
repeat  δ = −(H_gn(w) + λ·diag(H_gn(w)))⁻¹ ∇nlp(w), accept the step if it lowers `nlp`
(then λ /= 3) and otherwise increase λ (×10) and retry, until the relative decrease of `nlp` is
below `rel_tol`, the gradient norm is below `grad_tol·max(1, |nlp|)`, or `max_iters` steps.
`gauss_newton(w)` must return the (SPD) Gauss-Newton curvature of the same target, which the
multichain runners compute anyway for their mass matrix. Adam alone stalls short of the mode on
the large-K problems (b=7: the negative log posterior still oscillates by 10-30 % at the 30 000
iteration cap), which leaves the chains' warm start and the metric off the mode; a few damped
Gauss-Newton steps converge there because the target is a Gaussian likelihood of a smooth
residual.
"""
function refine_map_gauss_newton(w, nlp_and_grad, gauss_newton; max_iters = 50, λ0 = 1e-3,
                                 rel_tol = 1e-10, grad_tol = 1e-8, max_retries = 12)
    w = copy(w)
    nlp, grad = nlp_and_grad(w)
    history = Float64[nlp]
    λ = λ0
    iters = 0
    for _ in 1:max_iters
        norm(grad) <= grad_tol * max(1.0, abs(nlp)) && break
        H = Matrix(gauss_newton(w))
        d = diag(H)
        accepted = false
        for _ in 1:max_retries
            Hλ = copy(H)
            Hλ[diagind(Hλ)] .+= λ .* d
            δ = -(Symmetric(Hλ) \ grad)
            w_try = w .+ δ
            nlp_try, grad_try = nlp_and_grad(w_try)
            if isfinite(nlp_try) && nlp_try < nlp
                rel = (nlp - nlp_try) / max(1.0, abs(nlp))
                w, nlp, grad = w_try, nlp_try, grad_try
                λ = max(λ / 3, 1e-12)
                accepted = true
                iters += 1
                push!(history, nlp)
                rel < rel_tol && return w, iters, history
                break
            else
                λ *= 10
            end
        end
        accepted || break
    end
    return w, iters, history
end

"""
    predict_bpinn_stats(p, V, samples; α) -> (pred_u, pred_var)

Posterior predictive mean & variance of `u(x,t) = g(w·φ)` over `p`'s full `(x,t)` grid, accumulated
across posterior samples with Welford's online algorithm (avoids materializing `S × n_x × n_t`) --
the B-PINN analogue of `MPPINN.predict`/`MPPINN.predict_variance_grid`.
"""
function predict_bpinn_stats(p, V, samples; α = 0.9)
    n_x, n_t = length(p.x), length(p.tn)
    mean_grid = zeros(Float64, n_x, n_t)
    m2_grid = zeros(Float64, n_x, n_t)
    for (s, w) in enumerate(samples)
        w_raw = V * w
        u_grid = piecewise_activation.(predict_grid(p, w_raw), α)
        delta = u_grid .- mean_grid
        mean_grid .+= delta ./ s
        m2_grid .+= delta .* (u_grid .- mean_grid)
    end
    n = length(samples)
    var_grid = n > 1 ? m2_grid ./ (n - 1) : zeros(Float64, n_x, n_t)
    return mean_grid, var_grid
end

"""
    predict_residual_bpinn(p, V, w_mean; α) -> n_x × n_t matrix

Exact PDE residual `g'(w_raw·φ)·(w_raw·(∂φ/∂t+c·∂φ/∂x))` at the posterior mean weight `w_mean`
(in the eigenbasis) -- same formula as `GDPINN.predict_residual_gd`/`MPPINN.predict_residual`.
"""
function predict_residual_bpinn(p, V, w_mean; α = 0.9)
    w_raw = V * w_mean
    return piecewise_activation_derivative.(predict_grid(p, w_raw), α) .*
           predict_residual_grid(p, w_raw)
end

end # module BPINN
