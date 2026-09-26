# Variational Inference baseline for ONE advection wave speed: the EXACT SAME single-hidden-unit
# architecture, 2D Fourier basis, and eigenbasis-decorrelated design matrices as `mp_pinn_model.jl`
# (via `build_graph`, called with the identical rng seed `mp_pinn_run_beta.jl`/`bpinn_run_beta_multichain.jl`
# use), but with `w`'s posterior approximated by mean-field Gaussian VI (`vi_model.jl`, Algorithm 2
# of Yang, Meng & Karniadakis 2021 (B-PINNs, J. Comput. Phys.)) instead of HMC or message passing. Isolates the same
# thing the other baselines do -- architecture and data held fixed, only the estimator differs.
#
# Usage: julia --project=. pde/advection/vi_run_beta.jl <wave_speed> [traj_idx] [out_path] [N_z] [M]
#
# Prediction uses the CLOSED-FORM mean-field predictive (`predict_vi_stats_analytic`): the
# variational posterior is a diagonal Gaussian over the same eigenbasis coordinates MP-PINN
# produces, so its predictive moments are the exact activation moments -- no Monte Carlo, the
# same deterministic path as `MPPINN.predict_variance_grid`. This is the fair inference-time
# comparison (the M-sample Algorithm-2 step is a slower estimator of the identical quantity).

include(joinpath(@__DIR__, "..", "vi_model.jl"))

using .VIPINN
using .VIPINN.BPINN
using .VIPINN.BPINN.GDPINN
using .VIPINN.BPINN.GDPINN.MPPINN
using Random
using Printf
using HDF5

WAVE_SPEED = parse(Float64, ARGS[1])
TRAJ_IDX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : -1
N_Z = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
M_PREDICT = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 600

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(ARTIFACTS_DIR, "vi_beta$(WAVE_SPEED)_idx$(TRAJ_IDX).h5")

# Same hyperparameters and construction as mp_pinn_run_beta.jl's defaults -- see that file for the
# rationale behind each; kept identical here so all baselines differ only in how `w` is fit.
TIME_FILL = 0.5
Kx = 5
RESONANCE_MARGIN = 3
N_IC = 1024
NC_X = 48
SIGMA_W = 1.0
BETA_IC = 0.02
BETA_PDE = 0.05
PW_ALPHA = 0.9

MPPINN.Factors.MSG_DEBUG[] = 0

u, x, t = load_slice(; wave_speed = WAVE_SPEED, idx = TRAJ_IDX)
c_norm = estimate_c_norm(t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
Kt = ceil(Int, Kx * abs(c_norm)) + RESONANCE_MARGIN
println("[beta=$WAVE_SPEED] c_norm=$c_norm -> Kt=$Kt (VI baseline: N_z=$N_Z, M=$M_PREDICT)")

basis = TrigBasis(Kx, Kt)
p = Problem(basis, u, x, t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)

# Same rng seed as the other runners' build_graph call, so every baseline sees literally the same
# IC points, collocation points, and eigenbasis V -- only the estimator differs.
rng = MersenneTwister(7)
graph = build_graph(p, rng; n_ic = N_IC, nt = Kt, nx = NC_X, σw = SIGMA_W, β_ic = BETA_IC,
    β_pde = BETA_PDE, α = PW_ALPHA)
Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde = design_matrices(graph)
K = size(graph.V, 2)
println("[beta=$WAVE_SPEED] K=$(dim(basis)) raw features -> $K kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual points")

# ζ's initial state (Algorithm 2 leaves this unspecified): mu0=0 matches every other baseline's
# zero-mean-prior-consistent starting point (MP-PINN's prior, GD-PINN's w0, B-PINN's w0 before
# MAP); rho0=-3 gives an initial std of softplus(-3)~0.049, a small but nonzero starting spread.
μ0 = zeros(K)
ρ0 = fill(-3.0, K)
train_time_seconds = @elapsed begin
    μ, ρ, iters, loss_history = train_vi!(μ0, ρ0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA;
        N_z = N_Z, rng = MersenneTwister(21))
end
println(@sprintf("[beta=%s] VI: %d iters in %.1fs, final elbo_loss=%.4e",
    WAVE_SPEED, iters, train_time_seconds, loss_history[end]))

predict_time_seconds = @elapsed begin
    w = graph.V * μ  # rotated back out of the eigenbasis into raw trig coefficients
    pred_u, pred_var = predict_vi_stats_analytic(p, graph.V, μ, ρ; α = PW_ALPHA)
end
println(@sprintf("[beta=%s] predicted (analytic) in %.2fs", WAVE_SPEED, predict_time_seconds))

h5open(OUT_PATH, "w") do f
    attributes(f)["beta"] = WAVE_SPEED
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["Kx"] = basis.Kx
    attributes(f)["Kt"] = basis.Kt
    attributes(f)["c_norm"] = c_norm
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = VIPINN.BPINN.GDPINN.Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["N_z"] = N_Z
    attributes(f)["M_predict"] = M_PREDICT   # retained for the loader; unused by the analytic predict
    attributes(f)["predict_mode"] = "analytic"
    attributes(f)["iters"] = iters
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)
    # Variational posterior mean/std PER EIGENBASIS COORDINATE (not rotated back out via V) -- the
    # same coordinates MP-PINN's/B-PINN's `w_eigen_mean`/`w_eigen_std` are saved in (all runners
    # build `graph` with the identical rng seed), so directly comparable index-for-index. Unlike
    # B-PINN's empirical (sampling-noise-bearing) std, this is VI's own closed-form variational std
    # `softplus(ρ)` -- exact given the mean-field Gaussian family, no Monte Carlo noise.
    write(f, "w_eigen_mean", μ)
    write(f, "w_eigen_std", vi_std(ρ))
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "pred_var", pred_var)
    write(f, "loss_x", collect(1:length(loss_history)))
    write(f, "loss_y", loss_history)
end
println("[beta=$WAVE_SPEED] saved to $OUT_PATH")
