"""
Gradient-descent baseline for the SAME single-hidden-unit architecture `u = g(w·φ)` and 2D
Fourier basis `mp_pinn_model.jl` uses, trained by conventional Adam optimization on a PINN loss
(`MSE(u,y_ic)/β_ic² + MSE(R,0)/β_pde²`, `R = g'(z)·(w·r)` -- the exact same residual formula
`ResidualFactor` uses) instead of message passing. Reuses `build_graph` unchanged (same rng
seeds, same basis, same eigenbasis `V`), pulling its design matrices straight out of the factors
it already built, so both models train on literally identical data; only the estimator (loopy EP
vs point-estimate gradient descent) differs.
"""
module GDPINN

include(joinpath(@__DIR__, "mp_pinn_model.jl"))

using .MPPINN
using .MPPINN.Factors
using LinearAlgebra
using Printf
using Optimisers

export design_matrices, reaction_strength, pinn_loss, train_gd!, predict_gd, predict_residual_gd

@inline function piecewise_activation_second_derivative(x::Float64, α::Float64)
    x > 0 ? Factors.PIECEWISE_BETA^2 * exp(Factors.PIECEWISE_BETA * x) :
            -α * Factors.PIECEWISE_BETA^2 * exp(-Factors.PIECEWISE_BETA * x)
end

"""
    design_matrices(graph) -> (Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde)

Pulls the design matrices `build_graph` already built for message passing straight out of each
factor's own `.φ`/`.r`/`.u_obs`/`.β` fields, so training here uses literally the same points,
basis, and eigenbasis rotation `V` -- not a re-derivation.
"""
function design_matrices(graph)
    K = length(graph.w_idx)
    n_ic = length(graph.data_factors)
    Φ_ic = Matrix{Float64}(undef, n_ic, K)
    y_ic = Vector{Float64}(undef, n_ic)
    @inbounds for (i, f) in enumerate(graph.data_factors)
        Φ_ic[i, :] = f.φ
        y_ic[i] = f.u_obs
    end

    n_r = length(graph.res_factors)
    Φ_r = Matrix{Float64}(undef, n_r, K)
    R_mat = Matrix{Float64}(undef, n_r, K)
    @inbounds for (i, f) in enumerate(graph.res_factors)
        Φ_r[i, :] = f.φ
        R_mat[i, :] = f.r
    end

    return Φ_ic, y_ic, graph.data_factors[1].β, Φ_r, R_mat, graph.res_factors[1].β
end

"""
    reaction_strength(graph) -> Float64

The normalized reaction coefficient ρ of the graph's residual factors (`ResidualFactor.ρ`): 0 for
advection, `ρ·T_basis` for the Fisher-KPP (reaction-diffusion) graphs of `pde/diffusion/`. Every
residual factor of one graph carries the same value.
"""
function reaction_strength(graph)
    ρ = graph.res_factors[1].ρ
    all(f -> f.ρ == ρ, graph.res_factors) || error("residual factors carry different reaction strengths")
    return ρ
end

"""
    pinn_loss(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, α) -> Float64

Conventional PINN loss `MSE(g(Φ_ic·w), y_ic)/β_ic² + MSE(g'(Φ_r·w)·(R_mat·w), 0)/β_pde²` -- the
same data/PDE noise scales `DataObsFactor`/`ResidualFactor` use, reinterpreted as loss weights.
"""
function pinn_loss(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, α; reaction = 0.0)
    resid_ic = piecewise_activation.(Φ_ic * w, α) .- y_ic
    z_r = Φ_r * w
    R_r = piecewise_activation_derivative.(z_r, α) .* (R_mat * w)
    if reaction != 0.0
        g_r = piecewise_activation.(z_r, α)
        R_r = R_r .- reaction .* g_r .* (1 .- g_r)
    end
    return sum(abs2, resid_ic) / (length(y_ic) * β_ic^2) + sum(abs2, R_r) / (size(Φ_r, 1) * β_pde^2)
end

# Analytic gradient of pinn_loss: chain rule through g for the data term, product rule through
# R=g'(z)·Q (z=Φ_r·w, Q=R_mat·w) for the PDE term. With `reaction` = ρ ≠ 0 (Fisher-KPP, the same
# model form as `ResidualFactor`: R = g'(z)·Q − ρ·g(z)·(1−g(z))), the reaction term adds
# −ρ·(1−2g)·g'·φ to ∂R/∂w. The `reaction = 0` path is the advection residual.
function pinn_loss_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, α; reaction = 0.0)
    n_ic, n_r = length(y_ic), size(Φ_r, 1)

    z_ic = Φ_ic * w
    resid_ic = piecewise_activation.(z_ic, α) .- y_ic
    gp_ic = piecewise_activation_derivative.(z_ic, α)

    z_r = Φ_r * w
    Q_r = R_mat * w
    gp_r = piecewise_activation_derivative.(z_r, α)
    gpp_r = piecewise_activation_second_derivative.(z_r, α)
    R_r = gp_r .* Q_r

    grad_data = (2 / (n_ic * β_ic^2)) .* (Φ_ic' * (resid_ic .* gp_ic))
    if reaction == 0.0
        loss = sum(abs2, resid_ic) / (n_ic * β_ic^2) + sum(abs2, R_r) / (n_r * β_pde^2)
        grad_pde = (2 / (n_r * β_pde^2)) .* (Φ_r' * (R_r .* gpp_r .* Q_r) .+ R_mat' * (R_r .* gp_r))
    else
        g_r = piecewise_activation.(z_r, α)
        R_r = R_r .- reaction .* g_r .* (1 .- g_r)
        dR_dz = gpp_r .* Q_r .- reaction .* (1 .- 2 .* g_r) .* gp_r   # ∂R/∂z at fixed Q
        loss = sum(abs2, resid_ic) / (n_ic * β_ic^2) + sum(abs2, R_r) / (n_r * β_pde^2)
        grad_pde = (2 / (n_r * β_pde^2)) .* (Φ_r' * (R_r .* dR_dz) .+ R_mat' * (R_r .* gp_r))
    end
    return loss, grad_data .+ grad_pde
end

"""
    train_gd!(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, α; max_iters, lr0, window, plateau_tol,
              decay_factor, max_decays, check_every) -> (w, iters, loss_history)

Trains `w` by Adam on `pinn_loss`, well-optimized rather than run for a fixed budget: a
ReduceLROnPlateau + early-stopping schedule, the standard way to fit a full-batch gradient
descent run to how long it actually needs. Every `window` iterations, compares the mean loss
over the current window against the previous one (a single step's loss is too noisy to signal
convergence -- Adam's own momentum makes it oscillate even near a minimum); on a plateau
(relative improvement below `plateau_tol`), halves the learning rate, and stops once that's
happened `max_decays` times with no further improvement. Mirrors `train!`'s "stop once the
marginal update stops mattering" stopping rule, adapted to gradient descent's noisier per-step
loss instead of a per-epoch max-KL summary.
"""
function train_gd!(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, α;
                    max_iters = 30_000, lr0 = 3e-4, window = 500, plateau_tol = 1e-3,
                    decay_factor = 0.5, max_decays = 4, check_every = 1000, reaction = 0.0)
    lr = lr0
    opt_state = Optimisers.setup(Optimisers.Adam(lr), w)
    loss_history = Float64[]
    prev_window_mean = Inf
    decays = 0
    iters_done = 0
    for iter in 1:max_iters
        loss, grad = pinn_loss_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, α; reaction)
        push!(loss_history, loss)
        opt_state, w = Optimisers.update!(opt_state, w, grad)
        iters_done = iter
        if iter % check_every == 0
            @printf("  iter %6d  loss %.6e  lr %.2e\n", iter, loss, lr)
            flush(stdout)
        end
        if iter % window == 0
            window_mean = sum(@view loss_history[(end - window + 1):end]) / window
            if window_mean > prev_window_mean * (1 - plateau_tol)
                decays >= max_decays && break
                decays += 1
                lr *= decay_factor
                Optimisers.adjust!(opt_state, lr)
            end
            prev_window_mean = window_mean
        end
    end
    return w, iters_done, loss_history
end

"""
    predict_gd(p, V, w; α) -> n_x × n_t matrix

`u = g(w_raw·φ)` on `p`'s full grid, `w_raw = V·w` rotated back out of the eigenbasis -- same
formula as `MPPINN.predict`, fed the gradient-descent point estimate instead of a message-passing
posterior mean.
"""
predict_gd(p, V, w; α = 0.9) = piecewise_activation.(predict_grid(p, V * w), α)

"""
    predict_residual_gd(p, V, w; α) -> n_x × n_t matrix

Exact PDE residual `g'(w_raw·φ)·(w_raw·(∂φ/∂t+c·∂φ/∂x))` on `p`'s full grid -- same formula as
`MPPINN.predict_residual`, fed the gradient-descent point estimate.
"""
function predict_residual_gd(p, V, w; α = 0.9)
    w_raw = V * w
    return piecewise_activation_derivative.(predict_grid(p, w_raw), α) .* predict_residual_grid(p, w_raw)
end

end # module GDPINN
