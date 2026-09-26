"""
Trains/predicts MPPINN for ONE trajectory of the REAL PDEBench 1D reaction-diffusion dataset
(`data/1D/ReactionDiffusion/Train/ReacDiff_Nu<nu>_Rho<rho>.hdf5`, 10,000 random-IC trajectories
per (nu, rho), x ∈ [0,1) periodic, t ∈ [0,1]) -- unlike `mp_pinn_reacdiff_gaussian_run.jl`,
which uses the IC of PDEBench's bespoke single-trajectory generator (deterministic Gaussian bump
on [0, 2π)). Same graph construction (`reacdiff_lib.jl`), same hyperparameter defaults.

The spatial bandwidth `Kx` is chosen per trajectory from the observed initial condition's own
spectrum (smallest K whose first K Fourier modes carry ≥ 99.9% of the IC's AC energy, clamped
to [5, 8], see the Kx cap note below) -- training-data-informed, never test-informed: dataset ICs
are random sinusoid superpositions whose bandwidth varies by trajectory, and a fixed small Kx would
silently underfit the IC for the rougher ones (the pre-activation g⁻¹(u) needs at least the IC's
own bandwidth to represent t=0).

Training runs the ρ-continuation ladder (`train_continuation!` in `reacdiff_lib.jl`) by default;
pass a trailing `false` for single-shot training (`train!`, no continuation).

Usage: julia --project=. pde/diffusion/mp_pinn_reacdiff_dataset_run.jl <nu> <rho> <traj_idx> [out_path] [sigma_w] [beta_ic] [beta_pde] [Kt] [nt] [Kx_override] [pre_prior_z] [pre_prior_sigma_z] [continuation]
"""

include(joinpath(@__DIR__, "..", "mp_pinn_model.jl"))

using .MPPINN
using .MPPINN.Factors
using .MPPINN.Factors.DistributionCollections
using .MPPINN.Factors.Gaussian
using Random
using Statistics
using Printf
using HDF5

NU = parse(Float64, ARGS[1])
RHO = parse(Float64, ARGS[2])
TRAJ_IDX = parse(Int, ARGS[3])
SIGMA_W = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0
BETA_IC = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.02
BETA_PDE = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.05

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 4 ? ARGS[4] :
    joinpath(ARTIFACTS_DIR, "mp_pinn_reacdiff_ds2_nu$(NU)_rho$(RHO)_idx$(TRAJ_IDX).h5")

Kt = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 15
# nt >= 2Kt+1 so the stratified collocation grid cannot alias the basis' highest time
# frequency (nt=Kt leaves the constraint aliased).
NT = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : 2Kt + 1
KX_OVERRIDE = length(ARGS) >= 10 ? parse(Int, ARGS[10]) : 0  # 0 = IC-spectrum rule
PRE_PRIOR_Z = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 0.0  # 0 = off
PRE_PRIOR_SIGMA_Z = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 5.0
CONTINUATION = length(ARGS) >= 13 ? parse(Bool, ARGS[13]) : true  # false = single-shot training (no continuation)
TIME_FILL = 0.5
PW_ALPHA = 0.9
N_IC = 512
NC_X = 48
# Kx cap 8: EP on the Fisher residual factor destabilizes for Kx >= ~12 (the second-derivative
# coefficients scale as (2πk)²; measured: Kx=16 -> rel_l2 ~0.99 with non-convergent KL on every
# trajectory whose IC-spectrum rule picked it, Kx=8 -> 0.033 on the same trajectory). A capped
# basis fits a slightly smoothed IC instead of diverging.
# (the rule itself, `kx_from_ic` with kx_min = 5, kx_max = 8, energy_keep = 0.999, lives in reacdiff_lib.jl)

Factors.MSG_DEBUG[] = 0

include(joinpath(@__DIR__, "reacdiff_lib.jl"))

prob = reacdiff_dataset_problem(NU, RHO, TRAJ_IDX; Kt = Kt, kx_override = KX_OVERRIDE, time_fill = TIME_FILL)
u, x, t, basis, p, Kx = prob.u, prob.x, prob.t, prob.basis, prob.p, prob.Kx
NU_NORM, RHO_NORM = prob.nu_norm, prob.rho_norm
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] Kx=$Kx ($(KX_OVERRIDE > 0 ? "override" : "IC-spectrum rule")), Kt=$Kt, " *
        "nu_norm=$NU_NORM rho_norm=$RHO_NORM")

rng = MersenneTwister(7)
graph = build_reacdiff_graph(p, NU_NORM, RHO_NORM, rng; n_ic = N_IC, nt = NT, nx = NC_X,
                              σw = SIGMA_W, β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA,
                              pre_prior_z = PRE_PRIOR_Z, pre_prior_sigma_z = PRE_PRIOR_SIGMA_Z)
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] K=$(dim(basis)) raw -> $(size(graph.V, 2)) kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual factors")

train_time_seconds = @elapsed begin
    if CONTINUATION
        rung_epochs, kl_history = train_continuation!(graph, RHO_NORM)
    else
        epochs_single, kl_history = train!(graph)
        rung_epochs = [epochs_single]
    end
end
epochs = sum(rung_epochs)
# Predictive mean (E[g(z)]) and variance from one closed-form moment pass. The timed
# block is the same work as the VI baseline's (`vi_run_reacdiff.jl`) and the advection runner's;
# the residual at the posterior-mean function (a point-estimate diagnostic) is computed OUTSIDE
# the timed block (inside it, the residual grid evaluation would inflate the inference time
# of Ours relative to VI).
predict_time_seconds = @elapsed begin
    post = posterior_parameters(graph)
    w = graph.V * post.μ_vec
    pred_u, pred_var = predict_moments_grid(graph.V, p, post.μ_vec, post.σ_vec .^ 2; α = PW_ALPHA)
end
residual = predict_reacdiff_residual(p, w, NU_NORM, RHO_NORM; α = PW_ALPHA)
m = metrics(pred_u, u)
@printf("[nu=%s rho=%s idx=%d] trained %d epochs (rungs %s) in %.1fs | %s | mean|residual|=%.3e\n",
        NU, RHO, TRAJ_IDX, epochs, join(rung_epochs, "+"), train_time_seconds, fmt(m),
        Statistics.mean(abs.(residual)))

h5open(OUT_PATH, "w") do f
    attributes(f)["nu"] = NU
    attributes(f)["rho"] = RHO
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["Kx"] = Kx
    attributes(f)["Kt"] = Kt
    attributes(f)["nt"] = NT
    attributes(f)["epochs"] = epochs
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["rel_l2"] = m.rel_l2
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["pre_prior_z"] = PRE_PRIOR_Z
    attributes(f)["pre_prior_sigma_z"] = PRE_PRIOR_SIGMA_Z
    attributes(f)["continuation"] = Int(CONTINUATION)
    attributes(f)["rung_epochs"] = rung_epochs  # single-shot runs store [epochs]
    attributes(f)["predict_mode"] = "analytic"   # pred_u = E[g(z)] and pred_var from one moment pass
    attributes(f)["nu_norm"] = NU_NORM
    attributes(f)["rho_norm"] = RHO_NORM
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = Factors.PIECEWISE_BETA
    write(f, "x", x); write(f, "t", t); write(f, "tn", p.tn)
    write(f, "true_u", u); write(f, "pred_u", pred_u); write(f, "pred_var", pred_var)
    write(f, "residual", residual)
    write(f, "w", w)
    # Posterior marginal mean/std PER EIGENBASIS COORDINATE (same coordinates as the HMC / VI
    # baselines built with the identical rng seed), plus the V fingerprint.
    write(f, "w_eigen_mean", post.μ_vec)
    write(f, "w_eigen_std", post.σ_vec)
    write(f, "v_fingerprint", v_fingerprint(graph.V))
    write(f, "loss_x", collect(1:length(kl_history))); write(f, "loss_y", kl_history)
end
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] saved to $OUT_PATH")
