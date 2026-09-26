# Gradient-descent (conventional PINN) baseline for ONE PDEBench-dataset Fisher-KPP trajectory:
# the advection baseline `pde/advection/mp_pinn_gd_baseline_run_beta.jl` on the graph of
# `mp_pinn_reacdiff_dataset_run.jl` (same trajectory, Kx rule, basis, collocation points,
# eigenbasis V via the same rng seed). Adam on the PINN loss
#   MSE(g(z_ic), y_ic)/β_ic² + MSE(R, 0)/β_pde²,  R = g'(z)·(w·r) − ρ_norm·g(z)·(1 − g(z)),
# the reacdiff residual factor's own model form (`pinn_loss(...; reaction)`), with the same
# plateau / early-stopping schedule as every other Adam-trained component of this codebase.
# Point estimate only (no predictive variance).
#
# Training protocol: the SAME ρ-continuation ladder as the one-pass EP runner and the VI baseline
# (s in {0.25, 0.5, 0.75, 1}, warm-started rungs), within the advection GD baseline's total budget
# of 30 000 Adam iterations (7 500 per rung, each rung a fresh `train_gd!` with its own plateau
# schedule). Applied uniformly to every cell so that all three iterative methods share one
# protocol. Pass a trailing `false` (arg 5) for the single-shot 30 000-iteration protocol.
#
# Usage: julia --project=. pde/diffusion/gd_run_reacdiff.jl <nu> <rho> <traj_idx> [out_path] [continuation]

include(joinpath(@__DIR__, "..", "gd_pinn_model.jl"))

using .GDPINN
using .GDPINN.MPPINN
using .GDPINN.MPPINN.Factors
using .GDPINN.MPPINN.Factors.DistributionCollections
using .GDPINN.MPPINN.Factors.Gaussian
using Random
using Printf
using Statistics
using HDF5

NU = parse(Float64, ARGS[1])
RHO = parse(Float64, ARGS[2])
TRAJ_IDX = parse(Int, ARGS[3])
CONTINUATION = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : true
S_GRID = [0.25, 0.5, 0.75, 1.0]
TOTAL_ITERS = 30_000

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 4 ? ARGS[4] :
    joinpath(ARTIFACTS_DIR, "gd_pinn_reacdiff_nu$(NU)_rho$(RHO)_idx$(TRAJ_IDX).h5")

SIGMA_W = 1.0
BETA_IC = 0.02
BETA_PDE = 0.05
Kt = 15
NT = 2Kt + 1
TIME_FILL = 0.5
PW_ALPHA = 0.9
N_IC = 512
NC_X = 48

Factors.MSG_DEBUG[] = 0

include(joinpath(@__DIR__, "reacdiff_lib.jl"))

prob = reacdiff_dataset_problem(NU, RHO, TRAJ_IDX; Kt = Kt, time_fill = TIME_FILL)
u, x, t, basis, p, Kx = prob.u, prob.x, prob.t, prob.basis, prob.p, prob.Kx
NU_NORM, RHO_NORM = prob.nu_norm, prob.rho_norm
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] Kx=$Kx (IC-spectrum rule), Kt=$Kt, nu_norm=$NU_NORM rho_norm=$RHO_NORM (gradient-descent baseline)")

rng = MersenneTwister(7)
graph = build_reacdiff_graph(p, NU_NORM, RHO_NORM, rng; n_ic = N_IC, nt = NT, nx = NC_X,
                              σw = SIGMA_W, β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA)
Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde = design_matrices(graph)
REACTION = reaction_strength(graph)
REACTION == RHO_NORM || error("graph reaction strength $REACTION != rho_norm $RHO_NORM")   # as in the HMC runner
K = size(graph.V, 2)
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] K=$(dim(basis)) raw -> $K kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual points")

w0 = zeros(K)

"The ρ-continuation ladder for gradient descent: one fresh `train_gd!` per rung, warm-started."
function train_gd_ladder(w0)
    w_eig = copy(w0)
    rung_iters = Int[]
    loss_history = Float64[]
    for s_r in S_GRID
        w_eig, it, lh = train_gd!(w_eig, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, PW_ALPHA;
            max_iters = TOTAL_ITERS ÷ length(S_GRID), reaction = s_r * REACTION)
        push!(rung_iters, it)
        append!(loss_history, lh)
        println(@sprintf("[nu=%s rho=%s idx=%d] GD rung s=%.2f: %d iters, loss %.4e", NU, RHO, TRAJ_IDX, s_r, it, lh[end]))
        flush(stdout)
    end
    return w_eig, sum(rung_iters), rung_iters, loss_history
end

train_time_seconds = @elapsed begin
    if CONTINUATION
        w_eig, iters, rung_iters, loss_history = train_gd_ladder(w0)
    else
        w_eig, iters, loss_history = train_gd!(w0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, PW_ALPHA; reaction = REACTION)
        rung_iters = [iters]
    end
end
predict_time_seconds = @elapsed begin
    pred_u = predict_gd(p, graph.V, w_eig; α = PW_ALPHA)
end
w = graph.V * w_eig
residual = predict_reacdiff_residual(p, w, NU_NORM, RHO_NORM; α = PW_ALPHA)
m = metrics(pred_u, u)
println(@sprintf("[nu=%s rho=%s idx=%d] trained %d iters in %.1fs, predicted in %.2fs | %s | mean|residual|=%.3e",
    NU, RHO, TRAJ_IDX, iters, train_time_seconds, predict_time_seconds, fmt(m), Statistics.mean(abs.(residual))))

h5open(OUT_PATH, "w") do f
    attributes(f)["nu"] = NU
    attributes(f)["rho"] = RHO
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["nu_norm"] = NU_NORM
    attributes(f)["rho_norm"] = RHO_NORM
    attributes(f)["Kx"] = Kx
    attributes(f)["Kt"] = Kt
    attributes(f)["nt"] = NT
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["iters"] = iters
    attributes(f)["continuation"] = Int(CONTINUATION)
    attributes(f)["rung_iters"] = rung_iters
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["rel_l2"] = m.rel_l2
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)
    write(f, "w_eigen", w_eig)
    write(f, "v_fingerprint", v_fingerprint(graph.V))
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "residual", residual)
    write(f, "loss_x", collect(1:length(loss_history)))
    write(f, "loss_y", loss_history)
end
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] saved to $OUT_PATH")
