# Mean-field Gaussian VI baseline (Bayes by Backprop, `vi_model.jl`) for ONE PDEBench-dataset
# Fisher-KPP trajectory: the advection baseline `pde/advection/vi_run_beta.jl` on the graph of
# `mp_pinn_reacdiff_dataset_run.jl` (same trajectory, Kx rule, basis, collocation points,
# eigenbasis V via the same rng seed, prior and noise levels). The target density is the exact
# posterior of the reacdiff factor graph's model (`neg_log_posterior_and_grad(...; reaction)`,
# see `bpinn_run_reacdiff_multichain.jl`), the same one HMC samples; only the estimator differs.
# Prediction is the closed-form mean-field predictive (`predict_vi_stats_analytic`).
#
# Training protocol: the SAME ρ-continuation ladder the one-pass EP runner uses
# (`train_continuation!`: reaction strengths s·ρ_norm for s in {0.25, 0.5, 0.75, 1}, each rung
# warm-started from the previous one), within the advection VI baseline's total budget of
# 30 000 Adam iterations (7 500 per rung, each rung a fresh `train_vi!` call with its own
# plateau schedule, as each rung of the EP ladder is a fresh `train!`). Single-shot VI at the full
# reaction strength collapses to the trivial u ≡ 0 branch on this dataset (rel-L2 0.97 vs the
# stored trajectory on nu=0.5, rho=1, idx 898: the same lock-in mechanism as for single-shot EP),
# so the ladder is applied uniformly to every cell, exactly as for Ours.
# At the easier cells the shortfall is mainly that the variational width starts
# 10^3-10^4 x wider than the posterior and cannot shrink within the budget (a slow crawl, not a lock-in);
# at rho = 10 cold-started VI does converge to the trivial u = 0 mode. Pass a trailing `false` (arg 7)
# for single-shot training (30 000 iterations at the full reaction strength, no continuation).
#
# Usage: julia --project=. pde/diffusion/vi_run_reacdiff.jl <nu> <rho> <traj_idx> [out_path] [N_z] [M] [continuation]

include(joinpath(@__DIR__, "..", "vi_model.jl"))

using .VIPINN
using .VIPINN.BPINN
using .VIPINN.BPINN.GDPINN
using .VIPINN.BPINN.GDPINN.MPPINN
using .VIPINN.BPINN.GDPINN.MPPINN.Factors
using .VIPINN.BPINN.GDPINN.MPPINN.Factors.DistributionCollections
using .VIPINN.BPINN.GDPINN.MPPINN.Factors.Gaussian
using Random
using Printf
using Statistics
using HDF5

NU = parse(Float64, ARGS[1])
RHO = parse(Float64, ARGS[2])
TRAJ_IDX = parse(Int, ARGS[3])
N_Z = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 8
M_PREDICT = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 600
CONTINUATION = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : true
S_GRID = [0.25, 0.5, 0.75, 1.0]      # the EP runner's ladder (reacdiff_lib.jl: train_continuation!)
TOTAL_ITERS = 30_000                 # the advection VI baseline's budget, split evenly over the rungs

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 4 ? ARGS[4] :
    joinpath(ARTIFACTS_DIR, "vi_reacdiff_nu$(NU)_rho$(RHO)_idx$(TRAJ_IDX).h5")

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
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] Kx=$Kx (IC-spectrum rule), Kt=$Kt, nu_norm=$NU_NORM rho_norm=$RHO_NORM (VI baseline: N_z=$N_Z)")

rng = MersenneTwister(7)
graph = build_reacdiff_graph(p, NU_NORM, RHO_NORM, rng; n_ic = N_IC, nt = NT, nx = NC_X,
                              σw = SIGMA_W, β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA)
Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde = design_matrices(graph)
REACTION = reaction_strength(graph)
REACTION == RHO_NORM || error("graph reaction strength $REACTION != rho_norm $RHO_NORM")   # as in the HMC runner
K = size(graph.V, 2)
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] K=$(dim(basis)) raw -> $K kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual points")

# Same initial state as the advection VI baseline: mu0 = 0, rho0 = -3 (initial std softplus(-3) ~ 0.049).
μ0 = zeros(K)
ρ0 = fill(-3.0, K)

"The ρ-continuation ladder for VI: one fresh `train_vi!` per rung, warm-started (μ, ρ)."
function train_vi_ladder(μ0, ρ0)
    μ, ρ = copy(μ0), copy(ρ0)
    rung_iters = Int[]
    loss_history = Float64[]
    for (i, s_r) in enumerate(S_GRID)
        μ, ρ, it, lh = train_vi!(μ, ρ, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA;
            N_z = N_Z, max_iters = TOTAL_ITERS ÷ length(S_GRID), rng = MersenneTwister(20 + i),
            reaction = s_r * REACTION)
        push!(rung_iters, it)
        append!(loss_history, lh)
        println(@sprintf("[nu=%s rho=%s idx=%d] VI rung s=%.2f: %d iters, elbo_loss %.4e", NU, RHO, TRAJ_IDX, s_r, it, lh[end]))
        flush(stdout)
    end
    return μ, ρ, sum(rung_iters), rung_iters, loss_history
end

train_time_seconds = @elapsed begin
    if CONTINUATION
        μ, ρ, iters, rung_iters, loss_history = train_vi_ladder(μ0, ρ0)
    else
        μ, ρ, iters, loss_history = train_vi!(μ0, ρ0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA;
            N_z = N_Z, rng = MersenneTwister(21), reaction = REACTION)
        rung_iters = [iters]
    end
end
println(@sprintf("[nu=%s rho=%s idx=%d] VI: %d iters (rungs %s) in %.1fs, final elbo_loss=%.4e",
    NU, RHO, TRAJ_IDX, iters, join(rung_iters, "+"), train_time_seconds, loss_history[end]))

predict_time_seconds = @elapsed begin
    w = graph.V * μ
    pred_u, pred_var = predict_vi_stats_analytic(p, graph.V, μ, ρ; α = PW_ALPHA)
end
residual = predict_reacdiff_residual(p, w, NU_NORM, RHO_NORM; α = PW_ALPHA)
m = metrics(pred_u, u)
println(@sprintf("[nu=%s rho=%s idx=%d] predicted (analytic) in %.2fs | %s | mean|residual|=%.3e",
    NU, RHO, TRAJ_IDX, predict_time_seconds, fmt(m), Statistics.mean(abs.(residual))))

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
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["N_z"] = N_Z
    attributes(f)["M_predict"] = M_PREDICT   # retained for symmetry with the advection loader; unused
    attributes(f)["predict_mode"] = "analytic"
    attributes(f)["iters"] = iters
    attributes(f)["continuation"] = Int(CONTINUATION)
    attributes(f)["rung_iters"] = rung_iters
    attributes(f)["rel_l2"] = m.rel_l2
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)
    write(f, "w_eigen_mean", μ)
    write(f, "w_eigen_std", vi_std(ρ))
    write(f, "v_fingerprint", v_fingerprint(graph.V))
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "pred_var", pred_var)
    write(f, "residual", residual)
    write(f, "loss_x", collect(1:length(loss_history)))
    write(f, "loss_y", loss_history)
end
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] saved to $OUT_PATH")
