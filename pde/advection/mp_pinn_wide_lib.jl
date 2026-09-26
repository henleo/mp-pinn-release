"""
Wide (H hidden units, fixed readout) version of the MP-PINN pipeline in `mp_pinn_model.jl`:

    u(x,t) = Σ_{h=1..H} a_h · g(w_h · φ(x,t))

Everything except the factor combination is reused unchanged from `MPPINN`: the 2D Fourier
basis, the collocation sampling (same seeds), `orthogonalize_basis` (one shared eigenbasis `V`
for all neurons), the `N(0, σw²)` prior per weight, the noise levels, and `train!`'s schedule
(random sweeps over all factors until the largest per-factor KL divergence drops below the
tolerance). The wide factors themselves live in `lib/factors/factors_pinn_wide.jl`.

`build_wide_graph(p, rng; H = 1, a = [1.0])` with the same `rng` as `build_graph` reproduces the
single-neuron graph exactly, and `train_wide!` then reproduces `train!` bit for bit (with H = 1
and single-threaded BLAS, `OPENBLAS_NUM_THREADS=1`, `mp_pinn_run_beta_wide.jl` reproduces the
artifact of `mp_pinn_run_beta.jl`).
"""
module MPPINNWide

include(joinpath(@__DIR__, "..", "mp_pinn_model.jl"))

using .MPPINN
using .MPPINN.Factors
using .MPPINN.Factors.Gaussian
using .MPPINN.Factors.DistributionCollections
using LinearAlgebra
using Random
using Printf
using ProgressMeter

export WideFactorGraph, build_wide_graph, train_wide!, wide_posterior_parameters,
       wide_predict_moments, wide_predict_residual, wide_preactivation_grids, readout_weights

"""
    readout_weights(H) -> Vector{Float64}

The fixed readout vector `a` per width used in the experiments: distinct values so that no two
neurons receive identical messages from the symmetric start (with identical `a_h` the neurons
would stay identical forever and the model would collapse to one neuron).
"""
function readout_weights(H::Int)
    H == 1 && return [1.0]
    H == 2 && return [1.0, -1.0]
    H == 3 && return [1.0, -1.0, 0.5]
    H == 4 && return [1.0, -1.0, 0.5, -0.5]
    throw(ArgumentError("no default readout vector for H = $H; pass `a` explicitly"))
end

"""
    WideFactorGraph

`V` (rotation back to the raw trig coefficients, shared by all neurons), the message-passing
state, `w_idx[h]` = the `K` weight variables of neuron `h`, the readout `a`, and one wide factor
per labeled / collocation point.
"""
struct WideFactorGraph
    V::Matrix{Float64}
    db_marginal::DistributionBag{Gaussian1D}
    db_msg::DistributionBag{Gaussian1D}
    w_idx::Vector{Vector{Int64}}
    a::Vector{Float64}
    data_factors::Vector{WideDataObsFactor}
    res_factors::Vector{WideResidualFactor}
end

"""
    build_wide_graph(p, rng; H, a, n_ic, nt, nx, σw, β_ic, β_pde, α) -> WideFactorGraph

`MPPINN.build_graph` with `H` neurons: same design matrices, same `orthogonalize_basis` call,
same collocation points (drawn from `rng` in the same order), same prior per weight; each point
gets one `WideDataObsFactor` / `WideResidualFactor` holding one single-neuron factor per neuron
(all sharing the point's `φ_w` and `r_w` vectors).
"""
function build_wide_graph(p::Problem, rng::AbstractRNG; H::Int = 1, a::Vector{Float64} = readout_weights(H),
                          n_ic = 1024, nt = p.basis.Kt, nx = 48,
                          σw = 1.0, β_ic = 0.02, β_pde = 0.05, α = 0.9)
    length(a) == H || throw(ArgumentError("length(a) = $(length(a)) must equal H = $H"))
    println("Start bulding a wide graph (H=$H, a=$a)")
    A_ic, y_ic = initial_condition_observations(p; n_ic)
    A_r = residual_observations(p, MersenneTwister(1); nt, nx)
    println("Start building a basis")
    V = orthogonalize_basis(A_ic, A_r; β_ic, β_pde)

    db_marginal = DistributionBag(Gaussian1DUniform())
    db_msg = DistributionBag(Gaussian1DUniform())
    K = size(V, 2)
    w_idx = [[add!(db_marginal) for _ in 1:K] for _ in 1:H]
    @showprogress desc = "Setting priors" for h in 1:H, k in w_idx[h]
        db_marginal[k] = Gaussian1DFromMeanVariance(0.0, σw^2)
    end

    data_factors = WideDataObsFactor[]
    res_factors = WideResidualFactor[]

    stride = max(1, length(p.x) ÷ n_ic)
    @showprogress desc = "IC factors" for (i, r) in enumerate(1:stride:length(p.x))
        φ, dφdt, dφdx = phi_and_derivs(p.basis, p.tn[1], p.xn[r])
        φ_w = V' * φ
        r_w = V' * (dφdt .+ p.c_norm .* dφdx)
        push!(data_factors, WideDataObsFactor(
            [DataObsFactor(db_marginal, db_msg, w_idx[h], φ_w, p.u[r, 1], β_ic, α) for h in 1:H], a))
        push!(res_factors, WideResidualFactor(
            [ResidualFactor(db_marginal, db_msg, w_idx[h], φ_w, r_w, β_pde, α) for h in 1:H], a))
    end
    @showprogress desc = "residual factors" for it in 1:nt, ix in 1:nx
        tt = (it - rand(rng)) / nt * p.time_fill
        xx = (ix - rand(rng)) / nx
        φ, dφdt, dφdx = phi_and_derivs(p.basis, tt, xx)
        φ_w = V' * φ
        r_w = V' * (dφdt .+ p.c_norm .* dφdx)
        push!(res_factors, WideResidualFactor(
            [ResidualFactor(db_marginal, db_msg, w_idx[h], φ_w, r_w, β_pde, α) for h in 1:H], a))
    end

    return WideFactorGraph(V, db_marginal, db_msg, w_idx, a, data_factors, res_factors)
end

"""
    wide_posterior_parameters(graph) -> (μ, σ)

Marginal mean/std per weight in the eigenbasis, as `K × H` matrices (column `h` = neuron `h`).
"""
function wide_posterior_parameters(graph::WideFactorGraph)
    H = length(graph.w_idx)
    K = length(graph.w_idx[1])
    μ = Matrix{Float64}(undef, K, H)
    σ = Matrix{Float64}(undef, K, H)
    for h in 1:H, (k, idx) in enumerate(graph.w_idx[h])
        μ[k, h] = Gaussian.mean(graph.db_marginal[idx])
        σ[k, h] = sqrt(Gaussian.variance(graph.db_marginal[idx]))
    end
    return (μ = μ, σ = σ)
end

"""
    train_wide!(graph; max_epochs, kl_tol, check_every, rng) -> (epochs, kl_history)

`MPPINN.train!`'s schedule, verbatim, over the wide factors: one randomized sweep over every
data + residual factor per epoch, stop once the largest per-factor KL divergence of an epoch
drops below `kl_tol`.
"""
function train_wide!(graph::WideFactorGraph; max_epochs = 300, kl_tol = 1e-5, check_every = 5,
                     rng = MersenneTwister(11))
    n_data = length(graph.data_factors)
    n_res = length(graph.res_factors)
    order = collect(1:(n_data + n_res))
    kl_history = Float64[]
    epochs_done = 0
    for epoch in 1:max_epochs
        maxkl = 0.0
        shuffle!(rng, order)
        for idx in order
            kl = if idx <= n_data
                f = graph.data_factors[idx]
                forward = Factors.wide_data_forward_message(f)        # forward: approximate the output
                Factors.wide_data_backward_update!(f, forward)        # backward: update the weights
            else
                f = graph.res_factors[idx - n_data]
                forward = Factors.wide_residual_forward_message(f)    # forward: approximate the residual
                Factors.wide_residual_backward_update!(f, forward)    # backward: update the weights
            end
            maxkl = max(maxkl, kl)
        end
        push!(kl_history, maxkl)
        epochs_done = epoch
        if epoch % check_every == 0 || maxkl < kl_tol
            @printf("  epoch %3d  max KL %.3e\n", epoch, maxkl)
            flush(stdout)
        end
        maxkl < kl_tol && break
    end
    return epochs_done, kl_history
end

"""
    wide_predict_moments(graph, p; α) -> (mean_grid, var_grid, neuron_means, neuron_vars)

Posterior-predictive mean and variance of `u = Σ_h a_h g(z_h)` over `p`'s full grid from the
mean-field posterior: `predict_moments_grid` per neuron (exact activation moments of a diagonal
Gaussian in the eigenbasis), then `mean = Σ_h a_h mean_h`, `var = Σ_h a_h² var_h` (independent
neurons). The per-neuron grids `a_h mean_h` and `a_h² var_h` are returned too (n_x × n_t × H).
"""
function wide_predict_moments(graph::WideFactorGraph, p::Problem; α = 0.9)
    post = wide_posterior_parameters(graph)
    H = length(graph.a)
    n_x, n_t = length(p.x), length(p.tn)
    neuron_means = Array{Float64}(undef, n_x, n_t, H)
    neuron_vars = Array{Float64}(undef, n_x, n_t, H)
    mean_grid = zeros(Float64, n_x, n_t)
    var_grid = zeros(Float64, n_x, n_t)
    for h in 1:H
        m_h, v_h = predict_moments_grid(graph.V, p, post.μ[:, h], post.σ[:, h] .^ 2; α)
        neuron_means[:, :, h] = graph.a[h] .* m_h
        neuron_vars[:, :, h] = graph.a[h]^2 .* v_h
        mean_grid .+= @view neuron_means[:, :, h]
        var_grid .+= @view neuron_vars[:, :, h]
    end
    return mean_grid, var_grid, neuron_means, neuron_vars
end

"""
    wide_preactivation_grids(graph, p) -> n_x × n_t × H array

Pre-activation `z_h = w_h · φ` of every neuron over the full grid at the posterior mean
(`predict_grid` per neuron). This is the neuron-usage diagnostic.
"""
function wide_preactivation_grids(graph::WideFactorGraph, p::Problem)
    post = wide_posterior_parameters(graph)
    H = length(graph.a)
    z = Array{Float64}(undef, length(p.x), length(p.tn), H)
    for h in 1:H
        z[:, :, h] = predict_grid(p, graph.V * post.μ[:, h])
    end
    return z
end

"""
    wide_predict_residual(graph, p; α) -> n_x × n_t matrix

PDE residual `Σ_h a_h g'(z_h) · (w_h·(∂φ/∂t + c ∂φ/∂x))` of the wide network at the posterior
mean (`MPPINN.predict_residual` per neuron, weighted by `a_h`).
"""
function wide_predict_residual(graph::WideFactorGraph, p::Problem; α = 0.9)
    post = wide_posterior_parameters(graph)
    res = zeros(Float64, length(p.x), length(p.tn))
    for h in eachindex(graph.a)
        w_h = graph.V * post.μ[:, h]
        res .+= graph.a[h] .* piecewise_activation_derivative.(predict_grid(p, w_h), α) .*
                predict_residual_grid(p, w_h)
    end
    return res
end

end # module MPPINNWide
