# Wide (H hidden units, fixed readout a) version of the PDEBench-dataset Fisher-KPP graph:
# `build_reacdiff_graph` (reacdiff_lib.jl) with one `WideDataObsFactor` / `WideResidualFactor`
# per point (lib/factors/factors_pinn_wide.jl, reaction term through the wide factor's own ρ),
# and the ρ-continuation trainer (`train_continuation!`) for wide graphs. With H = 1, a = (1)
# the wide graph reproduces the single-neuron reacdiff graph (same design, same V, same
# collocation points) and `train_continuation_wide!` its training to 1e-12.
#
# Expects `MPPINNWide` (which exports `WideFactorGraph`, `train_wide!` and the predictors) and
# `reacdiff_lib.jl` to be loaded by the including script.

"""
    build_reacdiff_wide_graph(p, ν, ρ, rng; H, a, n_ic, nt, nx, σw, β_ic, β_pde, α) -> WideFactorGraph

`build_reacdiff_graph` with `H` neurons: same design matrices, same `orthogonalize_basis` call
(noise-floor rank tolerance), same collocation points drawn from `rng` in the same order, same
prior per weight; every point gets one wide factor holding one single-neuron factor per neuron
(the residual neurons built with ρ = 0, the reaction strength `ρ` sits in the wide factor).
"""
function build_reacdiff_wide_graph(p::Problem, ν::Float64, ρ::Float64, rng::AbstractRNG;
                                    H::Int = 1, a::Vector{Float64} = readout_weights(H),
                                    n_ic = 1024, nt = p.basis.Kt, nx = 48, σw = 1.0, β_ic = 0.02,
                                    β_pde = 0.05, α = 0.9)
    length(a) == H || throw(ArgumentError("length(a) = $(length(a)) must equal H = $H"))
    A_ic, y_ic = initial_condition_observations(p; n_ic)
    A_r = reacdiff_residual_observations(p.basis, ν, MersenneTwister(1); nt, nx, time_fill = p.time_fill)
    V = orthogonalize_basis(A_ic, A_r; β_ic, β_pde, rtol = nothing)

    db_marginal = DistributionBag(Gaussian1DUniform())
    db_msg = DistributionBag(Gaussian1DUniform())
    K = size(V, 2)
    w_idx = [[add!(db_marginal) for _ in 1:K] for _ in 1:H]
    for h in 1:H, k in w_idx[h]
        db_marginal[k] = Gaussian1DFromMeanVariance(0.0, σw^2)
    end

    data_factors = WideDataObsFactor[]
    res_factors = WideResidualFactor[]

    stride = max(1, length(p.x) ÷ n_ic)
    for r in 1:stride:length(p.x)
        φ, dφdt, _, d2φdx2 = phi_and_dxx(p.basis, p.tn[1], p.xn[r])
        φ_w = V' * φ
        r_w = V' * (dφdt .- ν .* d2φdx2)
        push!(data_factors, WideDataObsFactor(
            [DataObsFactor(db_marginal, db_msg, w_idx[h], φ_w, p.u[r, 1], β_ic, α) for h in 1:H], a))
        push!(res_factors, WideResidualFactor(
            [ResidualFactor(db_marginal, db_msg, w_idx[h], φ_w, r_w, β_pde, α) for h in 1:H], a, ρ))
    end
    for it in 1:nt, ix in 1:nx
        tt = (it - rand(rng)) / nt * p.time_fill
        xx = (ix - rand(rng)) / nx
        φ, dφdt, _, d2φdx2 = phi_and_dxx(p.basis, tt, xx)
        φ_w = V' * φ
        r_w = V' * (dφdt .- ν .* d2φdx2)
        push!(res_factors, WideResidualFactor(
            [ResidualFactor(db_marginal, db_msg, w_idx[h], φ_w, r_w, β_pde, α) for h in 1:H], a, ρ))
    end

    return WideFactorGraph(V, db_marginal, db_msg, w_idx, a, data_factors, res_factors)
end

"""
    train_continuation_wide!(graph, rho_norm; s_grid, max_epochs, kl_tol) -> (epochs_per_rung, kl_history)

`train_continuation!` for a wide graph: the same fixed ladder of reaction strengths
`s·rho_norm`, `s ∈ s_grid`; rungs with `s < 1` rebuild each `WideResidualFactor` around the
scaled reaction strength KEEPING its neurons (and therefore their message slots) so the
messages carry across rungs; the final rung trains `graph`'s own factors.
"""
function train_continuation_wide!(graph::WideFactorGraph, rho_norm::Float64;
                                  s_grid = [0.25, 0.5, 0.75, 1.0], max_epochs = 300, kl_tol = 1e-5)
    epochs_per_rung = Int[]
    kl_history = Float64[]
    for s in s_grid
        res_s = s == 1.0 ? graph.res_factors :
            [WideResidualFactor(f.neurons, f.a, s * rho_norm) for f in graph.res_factors]
        rung = WideFactorGraph(graph.V, graph.db_marginal, graph.db_msg, graph.w_idx, graph.a,
                               graph.data_factors, res_s)
        epochs, kls = train_wide!(rung; max_epochs, kl_tol)
        push!(epochs_per_rung, epochs)
        append!(kl_history, kls)
    end
    return epochs_per_rung, kl_history
end

"""
    wide_predict_reacdiff_residual(graph, p, ν, rho_norm; α) -> n_x × n_t matrix

The wide model's own residual `Σ_h a_h g'(z_h)·(∂_t − ν ∂_xx) z_h − ρ u (1 − u)` at the
posterior mean, `u = Σ_h a_h g(z_h)` (`predict_reacdiff_residual` per neuron for the linear
part; the reaction on the summed output).
"""
function wide_predict_reacdiff_residual(graph::WideFactorGraph, p::Problem, ν::Float64, rho_norm::Float64; α = 0.9)
    post = wide_posterior_parameters(graph)
    lin = zeros(Float64, length(p.x), length(p.tn))
    u = zeros(Float64, length(p.x), length(p.tn))
    for h in eachindex(graph.a)
        w_h = graph.V * post.μ[:, h]
        lin .+= graph.a[h] .* (predict_reacdiff_residual(p, w_h, ν, 0.0; α))
        u .+= graph.a[h] .* Factors.piecewise_activation.(predict_grid(p, w_h), α)
    end
    return lin .- rho_norm .* u .* (1 .- u)
end
