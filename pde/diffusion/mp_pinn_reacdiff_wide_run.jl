# Trains/predicts the WIDE one-pass MP-PINN (u = Σ_h a_h g(w_h·φ), H hidden units, fixed readout
# a) for ONE PDEBench-dataset Fisher-KPP trajectory, with the same ρ-continuation schedule,
# priors, noise levels, seeds and Kx rule as the single-neuron runner
# (`mp_pinn_reacdiff_dataset_run.jl`); the advection counterpart is
# `pde/advection/mp_pinn_run_beta_wide.jl`. Saves the same datasets/attributes as the
# single-neuron reacdiff artifact plus the width-specific ones (H, a, per-neuron posteriors as
# K x H matrices, neuron-usage summaries). Loader: `load_wide` in `reacdiff_baselines_sweep.py`.
#
# Usage: julia --project=. pde/diffusion/mp_pinn_reacdiff_wide_run.jl <nu> <rho> <traj_idx> <out_path> <H> [a_1,a_2,...]

include(joinpath(@__DIR__, "..", "advection", "mp_pinn_wide_lib.jl"))

using .MPPINNWide
using .MPPINNWide.MPPINN
using .MPPINNWide.MPPINN.Factors
using .MPPINNWide.MPPINN.Factors.DistributionCollections
using .MPPINNWide.MPPINN.Factors.Gaussian
using Random
using Printf
using Statistics
using HDF5

NU = parse(Float64, ARGS[1])
RHO = parse(Float64, ARGS[2])
TRAJ_IDX = parse(Int, ARGS[3])
OUT_PATH = ARGS[4]
H = parse(Int, ARGS[5])
A_READOUT = length(ARGS) >= 6 ? parse.(Float64, split(ARGS[6], ",")) : readout_weights(H)
length(A_READOUT) == H || error("readout vector has $(length(A_READOUT)) entries, H = $H")

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
include(joinpath(@__DIR__, "reacdiff_wide_lib.jl"))

prob = reacdiff_dataset_problem(NU, RHO, TRAJ_IDX; Kt = Kt, time_fill = TIME_FILL)
u, x, t, basis, p, Kx = prob.u, prob.x, prob.t, prob.basis, prob.p, prob.Kx
NU_NORM, RHO_NORM = prob.nu_norm, prob.rho_norm
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX H=$H] Kx=$Kx (IC-spectrum rule), Kt=$Kt, nu_norm=$NU_NORM rho_norm=$RHO_NORM (a=$A_READOUT)")

rng = MersenneTwister(7)  # identical to the single-neuron runner -> same V / K / collocation points
graph = build_reacdiff_wide_graph(p, NU_NORM, RHO_NORM, rng; H, a = A_READOUT, n_ic = N_IC, nt = NT, nx = NC_X,
                                  σw = SIGMA_W, β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA)
K = size(graph.V, 2)
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX H=$H] K=$(dim(basis)) raw -> $K kept x $H neurons, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual factors")

train_time_seconds = @elapsed (rung_epochs, kl_history) = train_continuation_wide!(graph, RHO_NORM)
epochs = sum(rung_epochs)
predict_time_seconds = @elapsed begin
    post = wide_posterior_parameters(graph)
    w = graph.V * post.μ  # K_raw x H
    pred_u, pred_var, neuron_means, neuron_vars = wide_predict_moments(graph, p; α = PW_ALPHA)
end
residual = wide_predict_reacdiff_residual(graph, p, NU_NORM, RHO_NORM; α = PW_ALPHA)
m = metrics(pred_u, u)
println(@sprintf("[nu=%s rho=%s idx=%d H=%d] trained %d epochs (rungs %s) in %.1fs, predicted in %.2fs; %s | mean|residual|=%.3e",
    NU, RHO, TRAJ_IDX, H, epochs, join(rung_epochs, "+"), train_time_seconds, predict_time_seconds, fmt(m),
    Statistics.mean(abs.(residual))))

z_grids = wide_preactivation_grids(graph, p)
neuron_z_min = [minimum(@view z_grids[:, :, h]) for h in 1:H]
neuron_z_max = [maximum(@view z_grids[:, :, h]) for h in 1:H]
neuron_z_rms = [sqrt(Statistics.mean(abs2, @view z_grids[:, :, h])) for h in 1:H]
neuron_out_rms = [sqrt(Statistics.mean(abs2, @view neuron_means[:, :, h])) for h in 1:H]
neuron_var_mean = [Statistics.mean(@view neuron_vars[:, :, h]) for h in 1:H]
for h in 1:H
    println(@sprintf("  neuron %d (a=%+.2f): z in [%.2f, %.2f] (rms %.2f), rms(a_h E[g]) = %.4f, mean(a_h^2 Var[g]) = %.3e",
        h, A_READOUT[h], neuron_z_min[h], neuron_z_max[h], neuron_z_rms[h], neuron_out_rms[h], neuron_var_mean[h]))
end

h5open(OUT_PATH, "w") do f
    attributes(f)["nu"] = NU
    attributes(f)["rho"] = RHO
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["H"] = H
    attributes(f)["nu_norm"] = NU_NORM
    attributes(f)["rho_norm"] = RHO_NORM
    attributes(f)["Kx"] = Kx
    attributes(f)["Kt"] = Kt
    attributes(f)["nt"] = NT
    attributes(f)["epochs"] = epochs
    attributes(f)["rung_epochs"] = rung_epochs
    attributes(f)["continuation"] = 1
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["rel_l2"] = m.rel_l2
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["predict_mode"] = "analytic"
    write(f, "a", A_READOUT)
    write(f, "x", x); write(f, "t", t); write(f, "tn", p.tn)
    write(f, "w", w)
    write(f, "w_eigen_mean", post.μ)
    write(f, "w_eigen_std", post.σ)
    write(f, "v_fingerprint", v_fingerprint(graph.V))
    write(f, "true_u", u); write(f, "pred_u", pred_u); write(f, "pred_var", pred_var)
    write(f, "residual", residual)
    write(f, "neuron_z_min", neuron_z_min); write(f, "neuron_z_max", neuron_z_max)
    write(f, "neuron_z_rms", neuron_z_rms); write(f, "neuron_out_rms", neuron_out_rms)
    write(f, "neuron_var_mean", neuron_var_mean)
    write(f, "loss_x", collect(1:length(kl_history))); write(f, "loss_y", kl_history)
end
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX H=$H] saved to $OUT_PATH")
