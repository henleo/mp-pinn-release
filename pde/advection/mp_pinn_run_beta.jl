# Trains/predicts MPPINN (mp_pinn_model.jl) for ONE advection wave speed and
# saves (x, t, true_u, pred_u) plus the trained posterior (raw trig weights `w` and the
# basis/normalization parameters needed to rebuild derivatives) and
# timing/epoch metadata to an HDF5 file -- the per-beta worker `mp_pinn_beta_sweep.py` shells
# out to for its grid sweep, which reconstructs the PDE residual itself from the posterior
# (see `predict_residual`'s formula) instead of receiving a precomputed grid. Same construction
# and hyperparameter defaults as the model's own build_graph pipeline; traj_idx defaults to -1
# (last trajectory), matching the gradient-descent baseline's own default so both estimators
# reference the same ground-truth trajectory per beta.
#
# Usage: julia --project=. pde/advection/mp_pinn_run_beta.jl <wave_speed> [traj_idx] [out_path] [sigma_w] [beta_ic] [beta_pde]
#
# `sigma_w`/`beta_ic`/`beta_pde` (default hyperparameters 1.0/0.02/0.05) are
# independent absolute overrides for sensitivity/calibration studies. No post-hoc recalibration
# of the predictive variance is applied.

include(joinpath(@__DIR__, "..", "mp_pinn_model.jl"))

using .MPPINN
using .MPPINN.Factors
using Random
using Printf
using HDF5

WAVE_SPEED = parse(Float64, ARGS[1])
TRAJ_IDX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : -1
SIGMA_W = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
BETA_IC = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.02
BETA_PDE = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.05

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(ARTIFACTS_DIR, "mp_pinn_beta$(WAVE_SPEED)_idx$(TRAJ_IDX).h5")

TIME_FILL = 0.5
Kx = 5
RESONANCE_MARGIN = 3
N_IC = 1024
NC_X = 48
PW_ALPHA = 0.9

Factors.MSG_DEBUG[] = 0

u, x, t = load_slice(; wave_speed = WAVE_SPEED, idx = TRAJ_IDX)
c_norm = estimate_c_norm(t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
# Sized off c_norm's resonance line purely for efficiency (a much smaller basis to build/train
# than one generous constant covering every beta), not because the basis needs to sit on the
# resonance line to fit well -- decoupling Kt from c_norm entirely (fixed Kt=150
# for every beta) gives the same accuracy at beta=4/7, just at a far larger, slower raw
# basis. `orthogonalize_basis` itself DOES still need the wave speed (via `residual_observations`'s
# `\vr`) to correctly decorrelate -- see that function's docstring for the ablation showing
# alternatives collapse the fit -- so this Kt sizing is an efficiency choice layered on top of
# an already wave-speed-aware basis, not an attempt to keep the basis itself PDE-agnostic.
Kt = ceil(Int, Kx * abs(c_norm)) + RESONANCE_MARGIN
println("[beta=$WAVE_SPEED] c_norm=$c_norm -> Kt=$Kt (σw=$SIGMA_W, β_ic=$BETA_IC, β_pde=$BETA_PDE)")

basis = TrigBasis(Kx, Kt)
p = Problem(basis, u, x, t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
rng = MersenneTwister(7)

graph = build_graph(p, rng; n_ic = N_IC, nt = Kt, nx = NC_X, σw = SIGMA_W, β_ic = BETA_IC,
    β_pde = BETA_PDE, α = PW_ALPHA)
println("[beta=$WAVE_SPEED] K=$(dim(basis)) raw features -> $(size(graph.V, 2)) kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual factors")

train_time_seconds = @elapsed (epochs, kl_history) = train!(graph)
# Predictive mean AND variance from ONE closed-form moment pass over the diagonal eigenbasis
# posterior (mean = E[g(z)], not the plug-in g(E[z])). This block is work-for-work
# the same as the VI baseline's timed block (`vi_run_beta.jl`), so the inference-time
# comparison is like-for-like.
predict_time_seconds = @elapsed begin
    post = posterior_parameters(graph)
    w = graph.V * post.μ_vec  # rotated back out of the eigenbasis into raw trig coefficients
    pred_u, pred_var = predict_moments_grid(graph.V, p, post.μ_vec, post.σ_vec .^ 2; α = PW_ALPHA)
end
println(@sprintf("[beta=%s] trained %d epochs in %.1fs, predicted in %.2fs", WAVE_SPEED, epochs, train_time_seconds, predict_time_seconds))

h5open(OUT_PATH, "w") do f
    attributes(f)["beta"] = WAVE_SPEED
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["epochs"] = epochs
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    # Basis/normalization needed to rebuild `phi_and_derivs` in Python and reconstruct the PDE
    # residual `g'(z) * (w_t + c_norm*w_x)` from the trained posterior directly --
    # see `predict_residual` in mp_pinn_model.jl for the same formula.
    attributes(f)["Kx"] = basis.Kx
    attributes(f)["Kt"] = basis.Kt
    attributes(f)["c_norm"] = c_norm
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["predict_mode"] = "analytic"   # pred_u = E[g(z)] and pred_var from one moment pass
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)
    # Posterior marginal mean/std PER EIGENBASIS COORDINATE (i.e. NOT rotated back out via V,
    # unlike `w` above) -- the same coordinates bpinn_run_beta_multichain.jl's HMC samples live in (both
    # runners build `graph` with the identical rng seed, so `V`/K_kept match exactly), so these
    # are directly comparable index-for-index against B-PINN's `w_eigen_mean`/`w_eigen_std` for a
    # weight-space posterior-recovery comparison.
    write(f, "w_eigen_mean", post.μ_vec)
    write(f, "w_eigen_std", post.σ_vec)
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "pred_var", pred_var)
    write(f, "loss_x", collect(1:length(kl_history)))
    write(f, "loss_y", kl_history)
end
println("[beta=$WAVE_SPEED] saved to $OUT_PATH")
