# Controlled baseline for ONE advection wave speed: the EXACT SAME single-hidden-unit
# architecture, 2D Fourier basis, and eigenbasis-decorrelated design matrices as `mp_pinn_model.jl`
# (via `build_graph`, called with the identical rng seed `mp_pinn_run_beta.jl` uses), but trained
# by conventional Adam gradient descent on a PINN loss (`gd_pinn_model.jl`) instead of message
# passing. Isolates whether MP-PINN's advantage comes from the architecture/basis choice or from
# the inference method itself -- everything but the estimator is held fixed.
#
# Usage: julia --project=. pde/advection/mp_pinn_gd_baseline_run_beta.jl <wave_speed> [traj_idx] [out_path]

include(joinpath(@__DIR__, "..", "gd_pinn_model.jl"))

using .GDPINN
using .GDPINN.MPPINN
using Random
using Printf
using HDF5

WAVE_SPEED = parse(Float64, ARGS[1])
TRAJ_IDX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : -1

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(ARTIFACTS_DIR, "gd_pinn_beta$(WAVE_SPEED)_idx$(TRAJ_IDX).h5")

# Same hyperparameters and construction as mp_pinn_run_beta.jl's defaults -- see that file for
# the rationale behind each; kept identical here so the two models differ only in how `w` is fit.
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
println("[beta=$WAVE_SPEED] c_norm=$c_norm -> Kt=$Kt (gradient-descent baseline)")

basis = TrigBasis(Kx, Kt)
p = Problem(basis, u, x, t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)

# Same rng seed as mp_pinn_run_beta.jl's build_graph call, so both models see literally the same
# IC points, collocation points, and eigenbasis V -- the only thing that differs is the estimator.
rng = MersenneTwister(7)
graph = build_graph(p, rng; n_ic = N_IC, nt = Kt, nx = NC_X, σw = SIGMA_W, β_ic = BETA_IC,
    β_pde = BETA_PDE, α = PW_ALPHA)
Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde = design_matrices(graph)
K = size(graph.V, 2)
println("[beta=$WAVE_SPEED] K=$(dim(basis)) raw features -> $K kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual points")

w0 = zeros(K)  # matches MP-PINN's zero-mean prior -- same starting point, same parameterization
train_time_seconds = @elapsed begin
    w, iters, loss_history = train_gd!(w0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, PW_ALPHA)
end
predict_time_seconds = @elapsed begin
    pred_u = predict_gd(p, graph.V, w; α = PW_ALPHA)
end
println(@sprintf("[beta=%s] trained %d iters in %.1fs, predicted in %.2fs", WAVE_SPEED, iters, train_time_seconds, predict_time_seconds))

h5open(OUT_PATH, "w") do f
    attributes(f)["beta"] = WAVE_SPEED
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["iters"] = iters
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["Kx"] = basis.Kx
    attributes(f)["Kt"] = basis.Kt
    attributes(f)["c_norm"] = c_norm
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = GDPINN.Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", graph.V * w)  # rotated back out of the eigenbasis into raw trig coefficients
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "loss_x", collect(1:length(loss_history)))
    write(f, "loss_y", loss_history)
end
println("[beta=$WAVE_SPEED] saved to $OUT_PATH")
