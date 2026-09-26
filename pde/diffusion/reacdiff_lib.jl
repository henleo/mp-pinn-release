# Shared Fisher-KPP (reaction-diffusion) helpers for the MP-PINN runners:
# second-derivative basis evaluation, residual design matrix, factor-graph construction, the
# ρ-continuation trainer, and the dense residual grid. Shared so that the Gaussian-bump runner
# (`mp_pinn_reacdiff_gaussian_run.jl`) and the PDEBench-dataset runner build the identical
# graph.
#
# Expects `MPPINN` (and its `Factors`/`DistributionCollections`/`Gaussian` submodules) to be
# loaded by the including script.
#
# The PDEBench-dataset problem construction (data file, per-trajectory Kx rule, normalized
# coefficients, `Problem`) lives here too (`reacdiff_dataset_problem`), so that the one-pass EP
# runner (`mp_pinn_reacdiff_dataset_run.jl`) and the HMC / VI / gradient-descent baselines for
# the same dataset (`bpinn_run_reacdiff_multichain.jl`, `vi_run_reacdiff.jl`,
# `gd_run_reacdiff.jl`) build literally the same problem and, with the same rng seed, the same
# graph and eigenbasis V.

using HDF5
using NPZ
using Statistics

"Matches Python's `str(x)` for the float values the sweeps use."
reacdiff_fmt_val(x::Float64) = isinteger(x) ? string(Int(x)) * ".0" : string(x)

reacdiff_dataset_path(nu::Float64, rho::Float64) =
    joinpath(@__DIR__, "..", "..", "data", "1D", "ReactionDiffusion", "Train",
             "ReacDiff_Nu$(reacdiff_fmt_val(nu))_Rho$(reacdiff_fmt_val(rho)).hdf5")

"""
    load_reacdiff_trajectory(nu, rho, traj_idx) -> (u, x, t)

One stored PDEBench trajectory (`u` is n_x × n_t in Float64); `traj_idx` is 0-based, negative
values count from the end (same convention as `load_slice`).
"""
function load_reacdiff_trajectory(nu::Float64, rho::Float64, traj_idx::Int)
    u, x, t = h5open(reacdiff_dataset_path(nu, rho), "r") do f
        n_traj = size(f["tensor"], 3)
        j = traj_idx < 0 ? n_traj + 1 + traj_idx : traj_idx + 1
        Float64.(f["tensor"][:, :, j]), Float64.(read(f, "x-coordinate")), Float64.(read(f, "t-coordinate"))
    end
    return u, x, t[1:size(u, 2)]
end

"""
    kx_from_ic(u0; kx_min = 5, kx_max = 8, energy_keep = 0.999) -> Int

Smallest K (in [kx_min, kx_max]) whose Fourier modes 1..K carry ≥ `energy_keep` of the initial
condition's total non-DC spectral energy. IC-informed basis sizing, computed from t=0 data only
(training-data-informed, never test-informed). The cap 8 is the disclosed EP limit (Kx ≥ ~12
destabilizes the Fisher residual factor, see `mp_pinn_reacdiff_dataset_run.jl`).
"""
function kx_from_ic(u0::Vector{Float64}; kx_min::Int = 5, kx_max::Int = 8, energy_keep::Float64 = 0.999)
    n = length(u0)
    v = u0 .- Statistics.mean(u0)
    total = sum(abs2, v)  # Parseval: total non-DC energy = (2/n)·Σ_{k≥1}|U_k|² for real signals
    total <= 0 && return kx_min
    acc = 0.0
    for k in 1:(n ÷ 2 - 1)
        c = s = 0.0
        for j in 0:(n - 1)  # direct DFT mode k (n=1024, only ~Kx_max modes ever needed)
            sj, cj = sincos(2π * k * j / n)
            c += v[j + 1] * cj
            s += v[j + 1] * sj
        end
        acc += 2 * (c^2 + s^2) / n
        if acc / total >= energy_keep
            return clamp(k, kx_min, kx_max)
        end
        k >= 4 * kx_max && break  # far past the cap; no need to resolve the exact cutoff
    end
    return kx_max
end

"""
    reacdiff_dataset_problem(nu, rho, traj_idx; Kt = 15, kx_override = 0, time_fill = 0.5)
        -> (u, x, t, basis, p, Kx, nu_norm, rho_norm)

The shared problem construction of every PDEBench-dataset Fisher-KPP runner: trajectory,
per-trajectory Kx (IC-spectrum rule unless overridden), `TrigBasis(Kx, Kt)`, the `Problem` on the
periodic domain of length L = x-spacing × n_x, and the normalized coefficients
`nu_norm = ν·T_basis/L²`, `rho_norm = ρ·T_basis` with `T_basis = (t_end − t_0)/time_fill`.
"""
function reacdiff_dataset_problem(nu::Float64, rho::Float64, traj_idx::Int; Kt::Int = 15,
                                  kx_override::Int = 0, time_fill::Float64 = 0.5)
    u, x, t = load_reacdiff_trajectory(nu, rho, traj_idx)
    Kx = kx_override > 0 ? kx_override : kx_from_ic(u[:, 1])
    L = x[end] - x[1] + (x[2] - x[1])  # full periodic domain length (1.0 for this dataset)
    basis = TrigBasis(Kx, Kt)
    p = Problem(basis, u, x, t; wave_speed = 0.0, time_fill = time_fill, L = L)
    T_basis = (t[end] - t[1]) / time_fill
    return (u = u, x = x, t = t, basis = basis, p = p, Kx = Kx,
            nu_norm = nu * T_basis / L^2, rho_norm = rho * T_basis)
end

"""
    reacdiff_gaussian_problem(nu, rho; Kt = 15, kx_override = 0, time_fill = 0.5)

The same problem construction as `reacdiff_dataset_problem`, for the Gaussian-bump initial
condition of PDEBench's single-trajectory generator on [0, 2π) (`init_mode="react"`). The
trajectory is the float64 nominal-PDE reference written by `reacdiff_reference.py`
(`gaussian_bump_reference`, cached as `artifacts/reacdiff_ref/ref_gauss_nu*_rho*.npz`); the
runner only uses its t=0 row as training data. Normalized coefficients use L = 2π, so the
diffusion is (2π)² ≈ 40× milder than the dataset's at the same nominal ν.
"""
function reacdiff_gaussian_problem(nu::Float64, rho::Float64; Kt::Int = 15, kx_override::Int = 0,
                                   time_fill::Float64 = 0.5)
    path = joinpath(@__DIR__, "..", "..", "artifacts", "reacdiff_ref",
                    "ref_gauss_nu$(reacdiff_fmt_val(nu))_rho$(reacdiff_fmt_val(rho)).npz")
    isfile(path) || error("missing $path -- run: python pde/diffusion/reacdiff_reference.py gauss $nu $rho")
    z = NPZ.npzread(path)
    u = permutedims(Float64.(z["u_ref"]))  # (n_t, n_x) -> (n_x, n_t)
    x, t = Float64.(z["x"]), Float64.(z["t"])
    Kx = kx_override > 0 ? kx_override : kx_from_ic(u[:, 1])
    L = x[end] - x[1] + (x[2] - x[1])  # full periodic domain length (2π)
    basis = TrigBasis(Kx, Kt)
    p = Problem(basis, u, x, t; wave_speed = 0.0, time_fill = time_fill, L = L)
    T_basis = (t[end] - t[1]) / time_fill
    return (u = u, x = x, t = t, basis = basis, p = p, Kx = Kx,
            nu_norm = nu * T_basis / L^2, rho_norm = rho * T_basis)
end

"""
    v_fingerprint(V) -> Vector{Float64}

Column sums of the eigenbasis `V` (one number per kept coordinate). Every reacdiff artifact
stores it so that the figure scripts can assert that the methods they compare in eigenbasis
coordinates were built on the identical `V` (the sign of an eigenvector depends on the BLAS
environment, e.g. BLAS threading; a flipped sign flips the column sum).
"""
v_fingerprint(V::AbstractMatrix) = vec(sum(V, dims = 1))

function trig_second_deriv(K::Int, v::Vector{Float64})
    ddv = Vector{Float64}(undef, 2K + 1)
    ddv[1] = 0.0
    for k in 1:K
        factor = -(2π * k)^2
        ddv[2k] = factor * v[2k]
        ddv[2k + 1] = factor * v[2k + 1]
    end
    return ddv
end

function phi_and_dxx(b::TrigBasis, tn::Float64, xn::Float64)
    T, dT = trig_vals(b.Kt, tn)
    X, dX = trig_vals(b.Kx, xn)
    ddX = trig_second_deriv(b.Kx, X)
    return kron(T, X), kron(dT, X), kron(T, dX), kron(T, ddX)
end

function reacdiff_residual_observations(basis::TrigBasis, ν::Float64, rng::AbstractRNG;
                                         nt::Int, nx::Int = 48, time_fill::Float64)
    A = Matrix{Float64}(undef, nt * nx, dim(basis))
    i = 0
    for it in 1:nt, ix in 1:nx
        tt = (it - rand(rng)) / nt * time_fill
        xx = (ix - rand(rng)) / nx
        _, dφdt, _, d2φdx2 = phi_and_dxx(basis, tt, xx)
        A[i += 1, :] = dφdt .- ν .* d2φdx2
    end
    return A
end

function build_reacdiff_graph(p::Problem, ν::Float64, ρ::Float64, rng::AbstractRNG; n_ic = 1024,
                               nt = p.basis.Kt, nx = 48, σw = 1.0, β_ic = 0.02, β_pde = 0.05, α = 0.9,
                               pre_prior_z = 0.0, pre_prior_sigma_z = 5.0)
    A_ic, y_ic = initial_condition_observations(p; n_ic)
    A_r = reacdiff_residual_observations(p.basis, ν, MersenneTwister(1); nt, nx, time_fill = p.time_fill)
    # Noise-floor rank tolerance (rtol = nothing): the second-derivative design pushes λ_max so
    # high at ν_norm·(2πKx)² ≳ 2.5e4 that the relative cutoff deletes the spatially-constant
    # slow-time subspace and forces a u ≡ 0 collapse -- see orthogonalize_basis' comment.
    V = orthogonalize_basis(A_ic, A_r; β_ic, β_pde, rtol = nothing)

    db_marginal = DistributionBag(Gaussian1DUniform())
    db_msg = DistributionBag(Gaussian1DUniform())
    K = size(V, 2)
    w_idx = [add!(db_marginal) for _ in 1:K]
    for k in w_idx
        db_marginal[k] = Gaussian1DFromMeanVariance(0.0, σw^2)
    end

    data_factors = DataObsFactor[]
    res_factors = ResidualFactor[]

    stride = max(1, length(p.x) ÷ n_ic)
    for r in 1:stride:length(p.x)
        φ, dφdt, _, d2φdx2 = phi_and_dxx(p.basis, p.tn[1], p.xn[r])
        φ_w = V' * φ
        push!(data_factors, DataObsFactor(db_marginal, db_msg, w_idx, φ_w, p.u[r, 1], β_ic, α))
        push!(res_factors, ResidualFactor(db_marginal, db_msg, w_idx, φ_w, V' * (dφdt .- ν .* d2φdx2), β_pde, α; ρ))
    end
    for it in 1:nt, ix in 1:nx
        tt = (it - rand(rng)) / nt * p.time_fill
        xx = (ix - rand(rng)) / nx
        φ, dφdt, _, d2φdx2 = phi_and_dxx(p.basis, tt, xx)
        φ_w = V' * φ
        push!(res_factors, ResidualFactor(db_marginal, db_msg, w_idx, φ_w, V' * (dφdt .- ν .* d2φdx2), β_pde, α; ρ))

        if pre_prior_z != 0.0
            # Persistent preactivation prior at this collocation point, reapplied every epoch
            # (optional, off by default: meant to pull the belief off the trivial u = 0 fixed point
            # at high ρ; the runners use the ρ-continuation ladder `train_continuation!` instead).
            gp_at_target = Factors.piecewise_activation_derivative(pre_prior_z, α)
            u_equiv = Factors.piecewise_activation(pre_prior_z, α)
            β_for_z_noise = pre_prior_sigma_z * gp_at_target
            push!(data_factors, DataObsFactor(db_marginal, db_msg, w_idx, φ_w, u_equiv, β_for_z_noise, α))
        end
    end

    return NLFactorGraph(V, db_marginal, db_msg, w_idx, data_factors, res_factors)
end

"""
    train_continuation!(graph, rho_norm; s_grid, max_epochs, kl_tol)
        -> (epochs_per_rung, kl_history)

ρ-continuation (homotopy) training for the Fisher-KPP graph: run
`train!` through a fixed ladder of reaction strengths `s*rho_norm`, `s ∈ s_grid`,
carrying the posterior forward between rungs. At high `rho_norm` a cold start converges to the
trivial u ≡ 0 branch (the residual factors' reaction-term precision, `(rho_norm·β_ic/β_pde)²`
per z, locks the diffuse prior belief onto the g(0)=0 root before the IC data can pull it off);
each rung instead starts from the previous rung's posterior, already near the data branch.
The ladder is a fixed protocol applied uniformly to every cell -- at low `rho_norm` it lands on
the same fixed point as single-shot training, at ~2x the cost.

Rungs with `s < 1` rebuild each `ResidualFactor` around the scaled reaction strength via the
inner constructor, KEEPING its message slots (`msg_to_w`) so the messages -- and therefore the
marginals -- carry across rungs; the final `s = 1` rung trains `graph`'s own factors. Each rung
is a full `train!` (which seeds its own schedule rng, so every rung shuffles identically).
`kl_history` is the pooled per-epoch max-KL trace across rungs; `epochs_per_rung[i]` says where
rung `i` ends inside it.
"""
function train_continuation!(graph::NLFactorGraph, rho_norm::Float64;
                             s_grid = [0.25, 0.5, 0.75, 1.0], max_epochs = 300, kl_tol = 1e-5)
    epochs_per_rung = Int[]
    kl_history = Float64[]
    for s in s_grid
        res_s = s == 1.0 ? graph.res_factors :
            [ResidualFactor(f.db_m, f.db_g, f.w_idx, f.φ, f.r, f.β, f.α, s * rho_norm, f.msg_to_w)
             for f in graph.res_factors]
        rung = NLFactorGraph(graph.V, graph.db_marginal, graph.db_msg, graph.w_idx,
                             graph.data_factors, res_s)
        epochs, kls = train!(rung; max_epochs, kl_tol)
        push!(epochs_per_rung, epochs)
        append!(kl_history, kls)
    end
    return epochs_per_rung, kl_history
end

function predict_reacdiff_residual(p::Problem, w::Vector{Float64}, ν::Float64, rho_norm::Float64; α = 0.9)
    Xmat = reduce(vcat, [trig_vals(p.basis.Kx, xi)[1]' for xi in p.xn])
    Tmat = reduce(vcat, [trig_vals(p.basis.Kt, ti)[1]' for ti in p.tn])
    dTmat = reduce(vcat, [trig_vals(p.basis.Kt, ti)[2]' for ti in p.tn])
    ddXmat = reduce(vcat, [trig_second_deriv(p.basis.Kx, trig_vals(p.basis.Kx, xi)[1])' for xi in p.xn])

    W = reshape(w, nxb(p.basis), ntb(p.basis))
    z = Xmat * W * Tmat'
    zt = Xmat * W * dTmat'
    zxx = ddXmat * W * Tmat'
    gp = Factors.piecewise_activation_derivative.(z, α)
    u_pred = Factors.piecewise_activation.(z, α)
    return gp .* (zt .- ν .* zxx) .- rho_norm .* u_pred .* (1 .- u_pred)
end
