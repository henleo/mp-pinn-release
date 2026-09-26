# Trains/predicts the WIDE MP-PINN (`mp_pinn_wide_lib.jl`: u = Σ_h a_h g(w_h·φ), H hidden
# units with fixed readout weights a, the same one-pass EP schedule as `mp_pinn_run_beta.jl`)
# for ONE advection wave speed and trajectory, and saves the same datasets/attributes as
# `mp_pinn_run_beta.jl` plus the width-specific ones (H, a, per-neuron posteriors as K x H
# matrices, per-neuron usage summaries). The Python loader is `mp_pinn_wide_sweep.py`.
#
# Usage: julia --project=. pde/advection/mp_pinn_run_beta_wide.jl <wave_speed> <traj_idx> <out_path> <H> [a_1,a_2,...]
#
# `a` defaults to `readout_weights(H)`: H=1 -> (1); 2 -> (1,-1); 3 -> (1,-1,0.5);
# 4 -> (1,-1,0.5,-0.5). Everything else (basis sizing, sigma_w=1, beta_ic=0.02, beta_pde=0.05,
# alpha=0.9, the schedule and its tolerance, the seeds) is identical to the single-neuron
# runner; with H=1 this runner reproduces `mp_pinn_run_beta.jl`'s artifact exactly.

include(joinpath(@__DIR__, "mp_pinn_wide_lib.jl"))

using .MPPINNWide
using .MPPINNWide.MPPINN
using .MPPINNWide.MPPINN.Factors
using Random
using Printf
using Statistics
using HDF5

WAVE_SPEED = parse(Float64, ARGS[1])
TRAJ_IDX = parse(Int, ARGS[2])
OUT_PATH = ARGS[3]
H = parse(Int, ARGS[4])
A_READOUT = length(ARGS) >= 5 ? parse.(Float64, split(ARGS[5], ",")) : readout_weights(H)
length(A_READOUT) == H || error("readout vector has $(length(A_READOUT)) entries, H = $H")

SIGMA_W = 1.0
BETA_IC = 0.02
BETA_PDE = 0.05
TIME_FILL = 0.5
Kx = 5
RESONANCE_MARGIN = 3
N_IC = 1024
NC_X = 48
PW_ALPHA = 0.9

Factors.MSG_DEBUG[] = 0

u, x, t = load_slice(; wave_speed = WAVE_SPEED, idx = TRAJ_IDX)
c_norm = estimate_c_norm(t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
Kt = ceil(Int, Kx * abs(c_norm)) + RESONANCE_MARGIN
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX H=$H] c_norm=$c_norm -> Kt=$Kt (a=$A_READOUT, σw=$SIGMA_W, β_ic=$BETA_IC, β_pde=$BETA_PDE)")

basis = TrigBasis(Kx, Kt)
p = Problem(basis, u, x, t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
rng = MersenneTwister(7)  # identical to mp_pinn_run_beta.jl -> same V / K / collocation points

graph = build_wide_graph(p, rng; H, a = A_READOUT, n_ic = N_IC, nt = Kt, nx = NC_X, σw = SIGMA_W,
    β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA)
K = size(graph.V, 2)
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX H=$H] K=$(dim(basis)) raw features -> $K kept x $H neurons, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual factors")

train_time_seconds = @elapsed (epochs, kl_history) = train_wide!(graph)
predict_time_seconds = @elapsed begin
    post = wide_posterior_parameters(graph)
    w = graph.V * post.μ  # K_raw x H raw trig coefficients, one column per neuron
    pred_u, pred_var, neuron_means, neuron_vars = wide_predict_moments(graph, p; α = PW_ALPHA)
end
m = metrics(pred_u, u)
println(@sprintf("[beta=%s idx=%d H=%d] trained %d epochs in %.1fs, predicted in %.2fs; %s",
    WAVE_SPEED, TRAJ_IDX, H, epochs, train_time_seconds, predict_time_seconds, fmt(m)))

# Neuron-usage diagnostics: pre-activation range per neuron at the posterior mean, the RMS of
# each neuron's contribution a_h E[g(z_h)] to the predictive mean, and the grid-mean of its
# contribution a_h^2 Var[g(z_h)] to the predictive variance.
z_grids = wide_preactivation_grids(graph, p)
neuron_z_min = [minimum(@view z_grids[:, :, h]) for h in 1:H]
neuron_z_max = [maximum(@view z_grids[:, :, h]) for h in 1:H]
neuron_z_rms = [sqrt(mean(abs2, @view z_grids[:, :, h])) for h in 1:H]
neuron_out_rms = [sqrt(mean(abs2, @view neuron_means[:, :, h])) for h in 1:H]
neuron_var_mean = [mean(@view neuron_vars[:, :, h]) for h in 1:H]
for h in 1:H
    println(@sprintf("  neuron %d (a=%+.2f): z in [%.2f, %.2f] (rms %.2f), rms(a_h E[g]) = %.4f, mean(a_h^2 Var[g]) = %.3e",
        h, A_READOUT[h], neuron_z_min[h], neuron_z_max[h], neuron_z_rms[h], neuron_out_rms[h], neuron_var_mean[h]))
end

h5open(OUT_PATH, "w") do f
    attributes(f)["beta"] = WAVE_SPEED
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["H"] = H
    attributes(f)["epochs"] = epochs
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["rel_l2"] = m.rel_l2
    attributes(f)["Kx"] = basis.Kx
    attributes(f)["Kt"] = basis.Kt
    attributes(f)["c_norm"] = c_norm
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["predict_mode"] = "analytic"   # pred_u = sum_h a_h E[g(z_h)], pred_var = sum_h a_h^2 Var[g(z_h)]
    write(f, "a", A_READOUT)
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)                      # K_raw x H
    write(f, "w_eigen_mean", post.μ)      # K x H, eigenbasis coordinates (same V as the single-neuron runners)
    write(f, "w_eigen_std", post.σ)       # K x H
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "pred_var", pred_var)
    write(f, "loss_x", collect(1:length(kl_history)))
    write(f, "loss_y", kl_history)
    write(f, "neuron_z_min", neuron_z_min)
    write(f, "neuron_z_max", neuron_z_max)
    write(f, "neuron_z_rms", neuron_z_rms)
    write(f, "neuron_out_rms", neuron_out_rms)
    write(f, "neuron_var_mean", neuron_var_mean)
end
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX H=$H] saved to $OUT_PATH")
