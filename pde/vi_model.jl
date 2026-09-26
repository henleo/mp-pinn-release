"""
Variational Inference baseline (`VIPINN`): the SAME single-hidden-unit architecture `u = g(w·φ)`
as `bpinn_model.jl`/`mp_pinn_model.jl`, but with `w`'s posterior approximated by mean-field
Gaussian VI (Algorithm 2 of Yang, Meng & Karniadakis 2021 (B-PINNs, J. Comput. Phys.) --
"Bayes by Backprop", Blundell et al. 2015) instead of HMC or loopy EP. Reuses `BPINN.neg_log_posterior_and_grad` for
the `-log P(θ) - log P(D|θ)` term (the exact same target density the HMC baseline samples) and
adds the mean-field Gaussian entropy term via the reparameterization trick -- no autodiff needed,
consistent with the rest of this codebase.
"""
module VIPINN

include(joinpath(@__DIR__, "bpinn_model.jl"))

using .BPINN
using .BPINN.GDPINN
using .BPINN.GDPINN.MPPINN
using .BPINN.GDPINN.MPPINN.Factors
using LinearAlgebra
using Random
using Printf
using Optimisers

export elbo_and_grad, train_vi!, predict_vi_stats, predict_vi_stats_analytic, softplus, vi_std

# Numerically stable softplus/sigmoid -- `softplus(ρ)` is Algorithm 2's `ln(1+exp(ζ_ρ))`
# parametrization of the variational std (guarantees positivity without a hard constraint).
softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
sigmoid(x) = 1 / (1 + exp(-x))
vi_std(ρ) = softplus.(ρ)

"""
    elbo_and_grad(μ, ρ, Z, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α) -> (L, grad_μ, grad_ρ)

Algorithm 2's per-iteration loss and gradient. `Z`'s `N_z` columns are iid `N(0,I)` draws;
`θ^(j) = μ + softplus(ρ) ⊙ Z[:,j]` (the reparameterization trick -- `Z` sampled by the caller, held
fixed for one gradient step). `L(ζ) = (1/N_z) Σⱼ [ln Q(θ^(j);ζ) − ln P(θ^(j)) − ln P(D|θ^(j))]`,
the negative ELBO. `-ln P(θ) − ln P(D|θ)` is exactly `BPINN.neg_log_posterior_and_grad`'s `nlp`
(dropping only additive constants independent of `ζ`, which don't affect the gradient below); the
mean-field Gaussian entropy term `ln Q(θ;ζ) = -0.5K ln(2π) − Σᵢ ln(σᵢ) − 0.5‖z‖²` is added directly
(also dropping its constant `-0.5K ln(2π)`).

Gradients (standard "Bayes by Backprop", Blundell et al. 2015): since `θ` is reparameterized,
`∂L/∂μ = (1/N_z) Σⱼ ∇nlp(θ^(j))` (the entropy term has no `μ`-dependence once expressed via `z`),
and `∂L/∂ρᵢ = (1/N_z) Σⱼ [∇nlp(θ^(j))]ᵢ·zᵢ^(j)·sigmoid(ρᵢ) − sigmoid(ρᵢ)/σᵢ` (chain rule through
`σ=softplus(ρ)`, plus the entropy term's `-ln(σᵢ)` derivative). The gradient agrees with finite
differences on a synthetic quadratic target (holding `Z` fixed across the perturbation).
"""
function elbo_and_grad(μ, ρ, Z, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; reaction = 0.0)
    σ = vi_std(ρ)
    K, Nz = size(Z)
    grad_μ = zeros(K)
    grad_ρ_data = zeros(K)
    L = 0.0
    for j in 1:Nz
        θ = μ .+ σ .* view(Z, :, j)
        nlp, g = neg_log_posterior_and_grad(θ, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; reaction)
        L += nlp
        grad_μ .+= g
        grad_ρ_data .+= g .* view(Z, :, j)
    end
    L /= Nz
    grad_μ ./= Nz
    grad_ρ_data ./= Nz
    grad_ρ = grad_ρ_data .* sigmoid.(ρ) .- sigmoid.(ρ) ./ σ
    L = L - sum(log.(σ)) - 0.5 * sum(abs2, Z) / Nz
    return L, grad_μ, grad_ρ
end

"""
    train_vi!(μ0, ρ0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; N_z, max_iters, lr0, window,
              plateau_tol, decay_factor, max_decays, check_every, rng) -> (μ, ρ, iters, loss_history)

Algorithm 2's training loop (`for k=1,...,N do ... end for`, "Update ζ with gradient ∇_ζL(ζ) using
Adam optimizer"). Each iteration draws `N_z` fresh `z~N(0,I)` samples, computes `elbo_and_grad`,
and updates `μ`/`ρ` with Adam. The paper leaves `N` (the iteration count) unspecified; here it's the
same Adam + plateau/early-stopping schedule this codebase already uses for every other Adam-trained
component (`GDPINN.train_gd!`, `BPINN.find_map`), for consistency, rather than a fixed count.
"""
function train_vi!(μ0, ρ0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α;
                    N_z = 8, max_iters = 30_000, lr0 = 3e-4, window = 500, plateau_tol = 1e-3,
                    decay_factor = 0.5, max_decays = 4, check_every = 1000, rng = MersenneTwister(21),
                    reaction = 0.0)
    μ, ρ = copy(μ0), copy(ρ0)
    K = length(μ)
    lr = lr0
    opt_μ = Optimisers.setup(Optimisers.Adam(lr), μ)
    opt_ρ = Optimisers.setup(Optimisers.Adam(lr), ρ)
    loss_history = Float64[]
    prev_window_mean = Inf
    decays = 0
    iters_done = 0
    for iter in 1:max_iters
        Z = randn(rng, K, N_z)
        L, grad_μ, grad_ρ = elbo_and_grad(μ, ρ, Z, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, σw, α; reaction)
        push!(loss_history, L)
        opt_μ, μ = Optimisers.update!(opt_μ, μ, grad_μ)
        opt_ρ, ρ = Optimisers.update!(opt_ρ, ρ, grad_ρ)
        iters_done = iter
        if iter % check_every == 0
            @printf("  [VI] iter %6d  elbo_loss %.6e  lr %.2e\n", iter, L, lr)
            flush(stdout)
        end
        if iter % window == 0
            window_mean = sum(@view loss_history[(end - window + 1):end]) / window
            if window_mean > prev_window_mean * (1 - plateau_tol)
                decays >= max_decays && break
                decays += 1
                lr *= decay_factor
                Optimisers.adjust!(opt_μ, lr)
                Optimisers.adjust!(opt_ρ, lr)
            end
            prev_window_mean = window_mean
        end
    end
    return μ, ρ, iters_done, loss_history
end

"""
    predict_vi_stats(p, V, μ, ρ; α, M, rng) -> (pred_u, pred_var)

Algorithm 2's final step ("Sample {z^(j)}_{j=1}^M ... Calculate {ũ(x,θ^(j))}_{j=1}^M as samples of
u(x)"): draws `M` fresh posterior samples `θ^(j) = μ + softplus(ρ)⊙z^(j)` and evaluates
`u(x,t)=g(w_raw·φ)` for each, accumulating mean/variance with Welford's algorithm -- the VI
analogue of `BPINN.predict_bpinn_stats`.
"""
function predict_vi_stats(p, V, μ, ρ; α = 0.9, M = 600, rng = MersenneTwister(37))
    σ = vi_std(ρ)
    K = length(μ)
    n_x, n_t = length(p.x), length(p.tn)
    mean_grid = zeros(Float64, n_x, n_t)
    m2_grid = zeros(Float64, n_x, n_t)
    for j in 1:M
        θ = μ .+ σ .* randn(rng, K)
        w_raw = V * θ
        u_grid = piecewise_activation.(predict_grid(p, w_raw), α)
        delta = u_grid .- mean_grid
        mean_grid .+= delta ./ j
        m2_grid .+= delta .* (u_grid .- mean_grid)
    end
    var_grid = m2_grid ./ (M - 1)
    return mean_grid, var_grid
end

"""
    predict_vi_stats_analytic(p, V, μ, ρ; α) -> (pred_u, pred_var)

Closed-form mean-field predictive -- the fair inference-time analogue of MP-PINN's deterministic
path. The variational posterior is a diagonal Gaussian `N(μ, diag(softplus(ρ)²))` over the SAME
eigenbasis coordinates MP-PINN produces, so the exact predictive moments `E[g(z)]`, `Var[g(z)]`
are given by `MPPINN.predict_moments_grid` -- the identical closed form MP uses, with no Monte
Carlo. Algorithm 2's `M`-sample step (`predict_vi_stats`) estimates these same moments and is
retained only for validation; the mean-field family has no correlations, so nothing is lost by
evaluating the moments directly instead of sampling them.
"""
function predict_vi_stats_analytic(p, V, μ, ρ; α = 0.9)
    σ2 = vi_std(ρ) .^ 2
    return MPPINN.predict_moments_grid(V, p, μ, σ2; α)
end

end # module VIPINN
