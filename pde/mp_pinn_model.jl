"""
Basis, factor-graph construction, and training/prediction machinery for the single-hidden-unit
MP-PINN `u = g(w·φ)`: a truncated 2D Fourier basis in (x,t), IC + PDE-residual observations
built from `DataObsFactor`/`ResidualFactor` (`lib/factors/factors_pinn.jl`),
solved by loopy EP/BP in the design matrix's eigenbasis (`orthogonalize_basis`).
"""
module MPPINN

include(joinpath(@__DIR__, "../lib/factors/factors.jl"))
include(joinpath(@__DIR__, "../lib/data.jl"))

using .Factors
using .Factors.Gaussian
using .Factors.DistributionCollections
using .Data
using HDF5
using LinearAlgebra
using Statistics
using Random
using Printf
using ProgressMeter

export TrigBasis, nxb, ntb, dim, trig_vals, phi_and_derivs
export Problem, load_slice, estimate_c_norm
export initial_condition_observations, residual_observations, orthogonalize_basis
export NLFactorGraph, build_graph, posterior_parameters, predict_slice, train!, predict, predict_grid
export predict_residual, predict_residual_grid, predict_variance_grid, predict_moments_grid, predict_moments
export downsample_idx, metrics, fmt

"""
    load_slice(; wave_speed, idx) -> (u, x, t)
"""
function load_slice(; wave_speed, idx)
    filename = "1D_Advection_Sols_beta" * @sprintf("%.1f", wave_speed) * ".hdf5"
    path = Data.advection_data_path(filename)
    u, x, t = h5open(path, "r") do f
        # Python/h5py-style 0-based indexing (idx=-1 -> last trajectory), so this matches the
        # `val_batch_idx` convention of PDEBench's `PINNDataset1Dpde`
        # (pdebench/models/pinn/utils.py) -- Julia has no native equivalent. For a
        # NON-negative idx this needs +1 (Python's 0-based idx==k is Julia's 1-based idx==k+1);
        # WITHOUT the +1, any figure comparing against PDEBench at a shared positive idx (e.g.
        # combined_solution_grid.py) would silently compare two
        # DIFFERENT real trajectories against each other, one index apart.
        n_traj = size(f["tensor"], 3)
        j = idx < 0 ? n_traj + 1 + idx : idx + 1
        f["tensor"][:, :, j], read(f, "x-coordinate"), read(f, "t-coordinate")
    end
    u2 = Float64.(u)
    return u2, Float64.(x), Float64.(t)[1:size(u2, 2)]
end

"""
    estimate_c_norm(t; wave_speed, time_fill) -> c_norm
"""
function estimate_c_norm(t; wave_speed, time_fill)
    T_basis = (t[end] - t[1]) / time_fill
    return wave_speed * T_basis
end

struct TrigBasis
    Kx::Int
    Kt::Int
end
nxb(b::TrigBasis) = 2b.Kx + 1
ntb(b::TrigBasis) = 2b.Kt + 1
dim(b::TrigBasis) = nxb(b) * ntb(b)

"""
    trig_vals(K, u) -> (v, dv)
"""
function trig_vals(K::Int, u::Float64)
    v = Vector{Float64}(undef, 2K + 1)
    dv = Vector{Float64}(undef, 2K + 1)
    v[1] = 1.0
    dv[1] = 0.0
    for k in 1:K
        s, c = sincos(2π * k * u)
        v[2k] = s
        v[2k + 1] = c
        dv[2k] = 2π * k * c
        dv[2k + 1] = -2π * k * s
    end
    return v, dv
end

"""
    phi_and_derivs(b, tn, xn) -> (φ, dφ/dtn, dφ/dxn)
"""
function phi_and_derivs(b::TrigBasis, tn::Float64, xn::Float64)
    T, dT = trig_vals(b.Kt, tn)
    X, dX = trig_vals(b.Kx, xn)
    return kron(T, X), kron(dT, X), kron(T, dX)
end

"""
    Problem

`x`/`t` are the raw coordinates exactly as loaded from data -- never rescaled in place.
`xn`/`tn` are the normalized coordinates every basis evaluation actually uses (`TrigBasis`
assumes a period-1 spatial domain and a `[0, time_fill]` time domain): `tn=(t-t[1])/T_basis`
does this for time; `xn=x/L` does the analogous thing for space, needed when the data's
real spatial domain length `L` isn't already 1 (e.g. PDEBench's `ReactionDiffusionEq`, `x∈[0,2π]`).
`L` defaults to `1.0`, so `xn≡x` for data already on a period-1 domain (e.g. PDEBench advection).
"""
struct Problem
    basis::TrigBasis
    u::Matrix{Float64}
    x::Vector{Float64}
    xn::Vector{Float64}
    t::Vector{Float64}
    tn::Vector{Float64}
    c_norm::Float64
    time_fill::Float64
end

function Problem(b::TrigBasis, u, x, t; wave_speed, time_fill, L = 1.0)
    T_basis = (t[end] - t[1]) / time_fill
    tn = (t .- t[1]) ./ T_basis
    xn = x ./ L
    Problem(b, u, x, xn, t, tn, wave_speed * T_basis, time_fill)
end

"""
    initial_condition_observations(p; n_ic) -> (A_ic, y_ic)

Design matrix + labels for the t=0 slice -- the only labeled data in this PINN.
"""
function initial_condition_observations(p::Problem; n_ic = 1024)
    stride = max(1, length(p.x) ÷ n_ic)
    rows = 1:stride:length(p.x)
    A = Matrix{Float64}(undef, length(rows), dim(p.basis))
    y = Vector{Float64}(undef, length(rows))
    for (i, r) in enumerate(rows)
        φ, _, _ = phi_and_derivs(p.basis, p.tn[1], p.xn[r])
        A[i, :] = φ
        y[i] = p.u[r, 1]
    end
    return A, y
end

"""
    residual_observations(p, rng; nt, nx) -> A_r

Design matrix for the PDE residual `∂φ/∂t + c·∂φ/∂x`, at stratified random collocation points
spanning the full (x,t) grid. Used by `build_graph`, alongside
`initial_condition_observations`, to build the Gram matrix `orthogonalize_basis` diagonalizes.

Replacing these rows by plain feature evaluations `φ` (removing the wave-speed dependence from
the decorrelation step, keeping it only in `ResidualFactor` itself) collapses accuracy
(beta=0.4: rel_l2 0.165 -> 0.544; beta=7.0: the first-epoch max KL rises from ~40 to ~1.2e4).
Stacking the raw, un-weighted partials `∂φ/∂t`/`∂φ/∂x` separately (still derivative-based,
just not combined via `c`) does not help either (rel_l2 0.515 at beta=0.4) -- so it isn't
merely "needs derivative information": the
ill-conditioned direction the eigendecomposition needs to see is specifically the wave-speed-
weighted characteristic direction `∂/∂t + c·∂/∂x`, matching where the PDE residual constraint
itself concentrates its information. `nt` must match the basis' own `Kt` or the constraint is
aliased and the extra Kt buys nothing.
"""
function residual_observations(p::Problem, rng::AbstractRNG; nt::Int, nx::Int = 48)
    A = Matrix{Float64}(undef, nt * nx, dim(p.basis))
    i = 0
    for it in 1:nt, ix in 1:nx
        tt = (it - rand(rng)) / nt * p.time_fill
        xx = (ix - rand(rng)) / nx
        _, dφdt, dφdx = phi_and_derivs(p.basis, tt, xx)
        A[i += 1, :] = dφdt .+ p.c_norm .* dφdx
    end
    return A
end

# Eigenbasis of G = A_ic'A_ic/β_ic² + A_r'A_r/β_pde² (sampling grid + noise levels, AND the wave
# speed -- see `residual_observations`' docstring for why that dependence is necessary, not
# just convenient). Rotating w into it (w = V*w') is required for EP: in the raw basis G is so
# ill-conditioned that loopy BP converges smoothly to the WRONG fixed point.
# Directions with eigenvalue ≈ 0 carry zero posterior information under the 0-mean prior on w,
# so dropping them (below `rtol * λ_max`) changes no prediction, only removes dead weight
# columns the graph would otherwise build.
#
# `rtol = nothing` (opt-in) switches to a noise-floor rank tolerance:
# keep λ > noise_safety·eps·K·λ_max, the eigendecomposition's own numerical noise floor times
# a fixed safety factor. Rationale: "eigenvalue ≈ 0" must be judged on an ABSOLUTE scale,
# because the RELATIVE cutoff `1e-8·λ_max` rises with the residual design's scale
# (λ_max ~ (ν_norm(2πKx)²)²/β_pde²) and, for the Fisher problem at ν_norm·(2πKx)² ≳ 2.5e4,
# would delete the spatially-constant slow-time directions (Gram eigenvalue
# ~ N_ic/β_ic² ≈ 1.3e6, data-determined a million-fold above the prior) -- leaving u ≡ 0 as
# the only residual-feasible solution and forcing a "bandwidth collapse" of the prediction.
# Any threshold between the noise floor and the IC block scale behaves identically; nothing
# here is fitted to data. The advection runners use the relative default `rtol = 1e-8`.
function orthogonalize_basis(A_ic, A_r; β_ic = 0.02, β_pde = 0.05, rtol = 1e-8,
                             noise_safety = 100.0)
    K = size(A_ic, 2)
    G = zeros(Float64, K, K)
    LinearAlgebra.BLAS.syrk!('U', 'T', 1 / β_ic^2, A_ic, 1.0, G)
    LinearAlgebra.BLAS.syrk!('U', 'T', 1 / β_pde^2, A_r, 1.0, G)
    ef = eigen(Symmetric(G, :U))
    λmax = maximum(ef.values)
    thresh = rtol === nothing ? noise_safety * eps(Float64) * K * λmax : rtol * λmax
    keep = ef.values .> thresh
    return ef.vectors[:, keep]
end

# Basis value/derivative matrices shared by predict_grid and predict_residual_grid --
# rows index x (or t), columns index the trig basis' own coefficients.
function _basis_grid_matrices(p::Problem)
    Xmat = reduce(vcat, [trig_vals(p.basis.Kx, xi)[1]' for xi in p.xn])
    dXmat = reduce(vcat, [trig_vals(p.basis.Kx, xi)[2]' for xi in p.xn])
    Tmat = reduce(vcat, [trig_vals(p.basis.Kt, ti)[1]' for ti in p.tn])
    dTmat = reduce(vcat, [trig_vals(p.basis.Kt, ti)[2]' for ti in p.tn])
    return Xmat, dXmat, Tmat, dTmat
end

"""
    predict_grid(p, w) -> n_x × n_t matrix

Pre-activation `z = w·φ` over `p`'s full (x,t) grid.
"""
function predict_grid(p::Problem, w::Vector{Float64})
    Xmat, _, Tmat, _ = _basis_grid_matrices(p)
    W = reshape(w, nxb(p.basis), ntb(p.basis))
    return Xmat * W * Tmat'
end

"""
    predict_residual_grid(p, w) -> n_x × n_t matrix

`w·(∂φ/∂t + c·∂φ/∂x)` over `p`'s full (x,t) grid -- the same quantity `residual_observations`
builds row-by-row at collocation points, but densely over the whole grid.
"""
function predict_residual_grid(p::Problem, w::Vector{Float64})
    Xmat, dXmat, Tmat, dTmat = _basis_grid_matrices(p)
    W = reshape(w, nxb(p.basis), ntb(p.basis))
    dzdx = dXmat * W * Tmat'
    dzdt = Xmat * W * dTmat'
    return dzdt .+ p.c_norm .* dzdx
end

downsample_idx(n, max_n) = 1:max(1, n ÷ max_n):n

"""
    metrics(pred, truth; split = 0.8)
"""
function metrics(pred, truth; split = 0.8)
    nt = size(truth, 2)
    ratios = [norm(pred[:, j] .- truth[:, j]) / norm(truth[:, j]) for j in 1:nt]
    j_split = clamp(floor(Int, split * nt), 1, nt - 1)
    (
        rel_l2 = norm(pred .- truth) / norm(truth),
        nrmse = Statistics.mean(ratios),
        nrmse_early = Statistics.mean(ratios[1:j_split]),
        nrmse_late = Statistics.mean(ratios[(j_split + 1):end]),
    )
end

fmt(m) = @sprintf(
    "rel_l2 %.3e | nrmse %.3e (early %.3e, late %.3e)",
    m.rel_l2, m.nrmse, m.nrmse_early, m.nrmse_late
)

"""
    NLFactorGraph

Everything `train!`/`predict` need: the rotation `V` back to the raw
trig-coefficient basis, and the message-passing state itself
(`db_marginal`/`db_msg`/`w_idx`/`data_factors`/`res_factors`).
"""
struct NLFactorGraph
    V::Matrix{Float64}
    db_marginal::DistributionBag{Gaussian1D}
    db_msg::DistributionBag{Gaussian1D}
    w_idx::Vector{Int64}
    data_factors::Vector{DataObsFactor}
    res_factors::Vector{ResidualFactor}
end

"""
    build_graph(p, rng; n_ic, nt, nx, σw, β_ic, β_pde, α) -> graph

Factor graph construction for u = g(w*φ). Rotates into G's eigenbasis first
(`orthogonalize_basis`), then creates the weight variables and one
`DataObsFactor` + `ResidualFactor` per labeled IC point, plus one
`ResidualFactor` per unlabeled collocation point.
"""
function build_graph(p::Problem, rng::AbstractRNG; n_ic = 1024, nt = p.basis.Kt, nx = 48,
                      σw = 1.0, β_ic = 0.02, β_pde = 0.05, α = 0.9)
    println("Start bulding a graph")
    A_ic, y_ic = initial_condition_observations(p; n_ic)
    A_r = residual_observations(p, MersenneTwister(1); nt, nx)
    println("Start building a basis")
    V = orthogonalize_basis(A_ic, A_r; β_ic, β_pde)

    db_marginal = DistributionBag(Gaussian1DUniform())
    db_msg = DistributionBag(Gaussian1DUniform())
    K = size(V, 2)
    w_idx = [add!(db_marginal) for _ in 1:K]
    @showprogress desc = "Setting priors" for k in w_idx
        db_marginal[k] = Gaussian1DFromMeanVariance(0.0, σw^2)
    end

    data_factors = DataObsFactor[]
    res_factors = ResidualFactor[]

    stride = max(1, length(p.x) ÷ n_ic)
    @showprogress desc = "IC factors" for (i, r) in enumerate(1:stride:length(p.x))
        φ, dφdt, dφdx = phi_and_derivs(p.basis, p.tn[1], p.xn[r])
        φ_w = V' * φ
        push!(data_factors, DataObsFactor(db_marginal, db_msg, w_idx, φ_w, p.u[r, 1], β_ic, α))
        push!(res_factors, ResidualFactor(db_marginal, db_msg, w_idx, φ_w, V' * (dφdt .+ p.c_norm .* dφdx), β_pde, α))
    end
    @showprogress desc = "residual factors" for it in 1:nt, ix in 1:nx
        tt = (it - rand(rng)) / nt * p.time_fill
        xx = (ix - rand(rng)) / nx
        φ, dφdt, dφdx = phi_and_derivs(p.basis, tt, xx)
        φ_w = V' * φ
        push!(res_factors, ResidualFactor(db_marginal, db_msg, w_idx, φ_w, V' * (dφdt .+ p.c_norm .* dφdx), β_pde, α))
    end

    return NLFactorGraph(V, db_marginal, db_msg, w_idx, data_factors, res_factors)
end

"""
    posterior_parameters(graph) -> (μ_vec, σ_vec)

Marginal mean/std per weight, in the eigenbasis `graph` was built in.
"""
function posterior_parameters(graph::NLFactorGraph)
    (
        μ_vec = [Gaussian.mean(graph.db_marginal[k]) for k in graph.w_idx],
        σ_vec = [sqrt(Gaussian.variance(graph.db_marginal[k])) for k in graph.w_idx],
    )
end

"""
    predict_slice(p, graph, j; α) -> (mean_vec, std_vec)

Posterior mean *and* std of `u(t_j,·) = g(w·φ)` across `x`, read off jointly
from one Gaussian belief per point -- not a hand-derived propagation formula.
Every message update in this framework already returns a Gaussian; `u`'s
belief comes from running the *same* forward factors the graph is built
from, once each, seeded with `graph`'s current marginals: projection
(`ScalarProductFactorWithFixedY`, `z=w·φ`), then activation
(`PiecewiseActivationFunctionFactor`, `u=g(z)`, exact moments).
"""
function predict_slice(p::Problem, graph::NLFactorGraph, j::Int; α = 0.9)
    Tj = trig_vals(p.basis.Kt, p.tn[j])[1]
    n = length(p.x)
    means = Vector{Float64}(undef, n)
    stds = Vector{Float64}(undef, n)
    for (i, xi) in enumerate(p.xn)
        Xi = trig_vals(p.basis.Kx, xi)[1]
        φ_w = graph.V' * kron(Tj, Xi)

        z_idx = add!(graph.db_marginal)
        proj = ScalarProductFactorWithFixedY(graph.db_marginal, graph.db_msg, graph.w_idx, φ_w, z_idx)
        update_msg_to_z!(proj)

        u_idx = add!(graph.db_marginal)
        act = PiecewiseActivationFunctionFactor(graph.db_marginal, graph.db_msg, z_idx, u_idx, α)
        update_msg_to_y!(act)

        belief = graph.db_marginal[u_idx]
        means[i] = Gaussian.mean(belief)
        stds[i] = sqrt(Gaussian.variance(belief))
    end
    return means, stds
end

"""
    train!(graph; max_epochs, kl_tol, check_every, rng) -> (epochs, kl_history)

Training. Each epoch is one randomized sweep over every data + residual
factor; stops once the largest per-factor KL divergence that epoch drops
below `kl_tol`. `kl_history[e]` is that max KL divergence for epoch `e`.
"""
function train!(graph::NLFactorGraph; max_epochs = 300, kl_tol = 1e-5, check_every = 5, rng = MersenneTwister(11))
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
                forward = Factors.data_forward_message(f)        # forward: approximate the output
                Factors.data_backward_update!(f, forward)        # backward: update the weight
            else
                f = graph.res_factors[idx - n_data]
                forward = Factors.residual_forward_message(f)    # forward: approximate the residual
                Factors.residual_backward_update!(f, forward)    # backward: update the weight
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
    predict(graph, p; α) -> n_x × n_t matrix

PLUG-IN prediction u = g(w·φ) at the posterior mean of w (rotated back out of the eigenbasis,
w = V·μ_vec), evaluated on p's full data grid. This is g(E[z]) -- a point estimate, NOT the
posterior-predictive mean E[g(z)]. The runners report mean and variance together from
`predict_moments` (one closed-form pass); keep this for point-estimate uses such as
residuals of the mean function.
"""
function predict(graph::NLFactorGraph, p::Problem; α = 0.9)
    post = posterior_parameters(graph)
    w = graph.V * post.μ_vec
    return piecewise_activation.(predict_grid(p, w), α)
end

"""
    predict_residual(graph, p; α) -> n_x × n_t matrix

Exact PDE residual `u_t + c*u_x = g'(w·φ)·(w·(∂φ/∂t+c·∂φ/∂x))` of the trained network
(see `ResidualFactor`'s own docstring for the same formula), at the posterior mean of
w, evaluated on p's full data grid.
"""
function predict_residual(graph::NLFactorGraph, p::Problem; α = 0.9)
    post = posterior_parameters(graph)
    w = graph.V * post.μ_vec
    z = predict_grid(p, w)
    r_grid = predict_residual_grid(p, w)
    return piecewise_activation_derivative.(z, α) .* r_grid
end

"""
    predict_moments_grid(V, p, μ, σ2; α) -> (mean_grid, var_grid)

Closed-form posterior-predictive moments of `u(x,t) = g(w·φ)` over `p`'s full (x,t) grid for a
DIAGONAL posterior over the eigenbasis coordinates -- `w = V·a`, `a ~ N(μ, diag(σ2))`. At each grid
point `z = w·φ_w = a·(Vᵀφ) ~ N(A·μ, A²·σ2)`, and `(E[g(z)], Var[g(z)])` are the exact activation
moments (`piecewise_activation_moments`). Dense per-time-slice matrix products, no per-point
factors, no sampling.

Shared by `predict_variance_grid` (the message-passing posterior) and the VI baseline
(`VIPINN.predict_vi_stats_analytic`): both posteriors are diagonal Gaussians over these same
coordinates, so both get the IDENTICAL closed form -- the fair, deterministic inference path (the
VI posterior admits the same amortized evaluation as ours; sampling it would be an implementation
choice, not a method property).
"""
function predict_moments_grid(V, p::Problem, μ, σ2; α = 0.9)
    n_x, n_t = length(p.x), length(p.tn)
    K = length(μ)
    b = p.basis

    Xmat = reduce(vcat, [trig_vals(b.Kx, xi)[1]' for xi in p.xn])   # n_x × nxb
    # V columns reshaped as (nxb, ntb, K): kron(T, X) has T outer, X inner, matching
    # Julia's column-major reshape (same convention as `predict_grid`).
    V3 = reshape(V, nxb(b), ntb(b), K)

    mean_grid = Matrix{Float64}(undef, n_x, n_t)
    var_grid = Matrix{Float64}(undef, n_x, n_t)
    Vj = Matrix{Float64}(undef, nxb(b), K)
    for j in 1:n_t
        Tj = trig_vals(b.Kt, p.tn[j])[1]
        fill!(Vj, 0.0)
        @inbounds for it in 1:ntb(b)
            Tj[it] == 0.0 && continue
            @views Vj .+= Tj[it] .* V3[:, it, :]
        end
        A = Xmat * Vj                       # n_x × K: row i = φ_w at (x_i, t_j)
        μz = A * μ
        σz2 = (A .^ 2) * σ2
        @inbounds for i in 1:n_x
            e_u, var_u = piecewise_activation_moments(μz[i], sqrt(σz2[i]), α)
            mean_grid[i, j] = e_u
            var_grid[i, j] = var_u
        end
    end
    return mean_grid, var_grid
end

"""
    predict_variance_grid(graph, p; α) -> n_x × n_t matrix

Posterior VARIANCE of `u(x,t) = g(w·φ)` over `p`'s full (x,t) grid: the closed-form activation
moments of the message-passing posterior (`predict_moments_grid`), matching the forward-message
moments `predict_slice` computes to machine precision but without allocating one factor pair per
grid point (each `predict_slice` point permanently appends ~K message slots to the graph's
`DistributionBag`s, which grows a b=7 run to ~6 GiB RSS during prediction).
"""
function predict_variance_grid(graph::NLFactorGraph, p::Problem; α = 0.9)
    post = posterior_parameters(graph)
    _, var_grid = predict_moments_grid(graph.V, p, post.μ_vec, post.σ_vec .^ 2; α)
    return var_grid
end

"""
    predict_moments(graph, p; α) -> (mean_grid, var_grid)

Posterior-predictive MEAN and VARIANCE of `u = g(w·φ)` over `p`'s full grid from the
message-passing posterior, in ONE closed-form pass (`predict_moments_grid`). The mean is the
exact E[g(z)] -- not the plug-in g(E[z]) of `predict` -- so mean and variance come from the
same moment computation: the reported predictive is internally consistent and the
timed inference path is work-for-work identical to the VI baseline's
(`VIPINN.predict_vi_stats_analytic`).
"""
function predict_moments(graph::NLFactorGraph, p::Problem; α = 0.9)
    post = posterior_parameters(graph)
    return predict_moments_grid(graph.V, p, post.μ_vec, post.σ_vec .^ 2; α)
end

end # module MPPINN
