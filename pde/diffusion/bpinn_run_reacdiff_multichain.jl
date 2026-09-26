# Multi-chain, mass-preconditioned HMC reference for ONE PDEBench-dataset Fisher-KPP
# (reaction-diffusion) trajectory: the advection reference `pde/advection/bpinn_run_beta_multichain.jl`
# transplanted onto the graph of `mp_pinn_reacdiff_dataset_run.jl`. Same trajectory, same
# per-trajectory Kx rule, same basis, collocation points, eigenbasis V (same rng seed), prior and
# noise levels as the one-pass EP runner, so the posterior is over the SAME eigenbasis coordinates.
#
# Target density = the exact posterior of the model the reacdiff factor graph encodes:
#   0.5·Σ(g(z_ic) − y)²/β_ic² + 0.5·Σ R²/β_pde² + 0.5·‖w‖²/σw²,
#   R = g'(z)·(w·r) − ρ_norm·g(z)·(1 − g(z)),  r = V'(∂φ/∂t − ν_norm ∂²φ/∂x²),
# i.e. `ResidualFactor`'s own model form (which applies the operator through g' only and omits the
# g'' chain-rule diffusion term) -- the reference samples the model that EP and VI
# approximate, not the true PDE. `neg_log_posterior_and_grad(...; reaction = ρ_norm)`.
#
# Sampler (identical protocol to the advection reference): MAP warm start by Adam +
# damped Gauss-Newton polish; dense Gauss-Newton mass matrix at the MAP; n_chains chains (chain 1
# at the MAP, the others at 2σ posterior-shaped jittered points); per chain a Hoffman-Gelman
# initial step size, dual-averaging adaptation over the burn-in towards acceptance 0.8, then a
# fixed step for the retained draws; the number of leapfrog steps of every proposal is drawn
# uniformly from 1..L (Neal 2011, avoids the near-periodicity of a fixed trajectory length).
# Per-chain sample matrices are saved for split-R-hat / ESS.
#
# One Fisher-specific addition: the MAP search is a two-start search: (i) Adam from w = 0
# through the SAME ρ-continuation ladder as the EP / VI / GD runners (reaction strengths s·ρ_norm,
# s in {0.25, 0.5, 0.75, 1}, warm-started rungs, 30 000 iterations in total) and (ii) Adam at the
# full reaction strength from the one-pass EP posterior mean of the same cell (the cached
# `mp_pinn_reacdiff_ds2_*` artifact, same eigenbasis, checked through `v_fingerprint`); both are
# polished by damped Gauss-Newton and the lower negative log posterior wins. Reason: the exact
# posterior has a spurious local mode on the trivial u ≡ 0 branch, and Adam from w = 0 lands on
# it at high ρ_norm even through the ladder (ν=2, ρ=10, idx 898: rel-L2 0.999 of the MAP fit, nlp
# 2.4e5, against the data-branch mode found from the EP mean); the chains then
# sample the mode with the (much) lower negative log posterior, which is the one every method is
# compared on. The artifact records both candidates' nlp, which start won, and the MAP's rel-L2
# against the stored trajectory (`map_rel_l2`). Starting a local optimizer from another method's
# solution only selects the mode; the MAP and the chains are those of the exact posterior.
#
# Usage: julia --project=. pde/diffusion/bpinn_run_reacdiff_multichain.jl <nu> <rho> <traj_idx> [out_path]
#            [num_samples] [burn] [L] [n_chains] [jitter]

include(joinpath(@__DIR__, "..", "bpinn_model.jl"))

using .BPINN
using .BPINN.GDPINN
using .BPINN.GDPINN.MPPINN
using .BPINN.GDPINN.MPPINN.Factors
using .BPINN.GDPINN.MPPINN.Factors.DistributionCollections
using .BPINN.GDPINN.MPPINN.Factors.Gaussian
using .BPINN.GDPINN.MPPINN.Factors: piecewise_activation, piecewise_activation_derivative
using Random
using Printf
using Statistics
using LinearAlgebra
using HDF5

NU = parse(Float64, ARGS[1])
RHO = parse(Float64, ARGS[2])
TRAJ_IDX = parse(Int, ARGS[3])
NUM_SAMPLES = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1000
BURN = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 250
LEAPFROG_L = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 20
N_CHAINS = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 4
JITTER = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 2.0
TARGET_ACCEPT = 0.8

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 4 ? ARGS[4] :
    joinpath(ARTIFACTS_DIR, "bpinn_mc_reacdiff_nu$(NU)_rho$(RHO)_idx$(TRAJ_IDX).h5")

# Same hyperparameters as mp_pinn_reacdiff_dataset_run.jl's defaults.
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
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] Kx=$Kx (IC-spectrum rule), Kt=$Kt, nu_norm=$NU_NORM rho_norm=$RHO_NORM " *
        "(multichain HMC: $N_CHAINS chains x $NUM_SAMPLES samples, burn=$BURN, L=$LEAPFROG_L)")

rng = MersenneTwister(7)  # identical to the one-pass runner -> same V / K / collocation points
graph = build_reacdiff_graph(p, NU_NORM, RHO_NORM, rng; n_ic = N_IC, nt = NT, nx = NC_X,
                              σw = SIGMA_W, β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA)
Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde = design_matrices(graph)
REACTION = reaction_strength(graph)
REACTION == RHO_NORM || error("graph reaction strength $REACTION != rho_norm $RHO_NORM")
K = size(graph.V, 2)
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] K=$(dim(basis)) raw -> $K kept, " *
        "$(length(graph.data_factors)) labeled (IC), $(length(graph.res_factors)) residual points")

nlp_and_grad(w) = neg_log_posterior_and_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA;
                                             reaction = REACTION)

S_GRID = [0.25, 0.5, 0.75, 1.0]   # the runners' shared ρ-continuation ladder (reacdiff_lib.jl)
TOTAL_MAP_ITERS = 30_000

"Adam MAP search through the ρ-continuation ladder: one fresh `find_map` per rung, warm-started."
function find_map_ladder(w0)
    w = copy(w0)
    rung_iters = Int[]
    for s_r in S_GRID
        w, it, _ = find_map(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA;
                            max_iters = TOTAL_MAP_ITERS ÷ length(S_GRID), reaction = s_r * REACTION)
        push!(rung_iters, it)
        println(@sprintf("[nu=%s rho=%s idx=%d] MAP rung s=%.2f: %d Adam iters", NU, RHO, TRAJ_IDX, s_r, it))
        flush(stdout)
    end
    return w, rung_iters
end

w0 = zeros(K)
map_time_seconds = @elapsed begin
    w_map_zero, map_rung_iters = find_map_ladder(w0)
end
map_iters = sum(map_rung_iters)
println(@sprintf("[nu=%s rho=%s idx=%d] MAP start (i), Adam ladder from zero: %d iters (rungs %s) in %.1fs, nlp %.6e",
    NU, RHO, TRAJ_IDX, map_iters, join(map_rung_iters, "+"), map_time_seconds, nlp_and_grad(w_map_zero)[1]))

# Start (ii): the one-pass EP posterior mean of the same cell, if cached on the same eigenbasis.
EP_ARTIFACT = joinpath(ARTIFACTS_DIR, "mp_pinn_reacdiff_ds2_nu$(NU)_rho$(RHO)_idx$(TRAJ_IDX).h5")
w_map_ep = nothing
map_ep_iters = 0
if isfile(EP_ARTIFACT)
    μ_ep, fp_ep = h5open(f -> (read(f, "w_eigen_mean"), haskey(f, "v_fingerprint") ? read(f, "v_fingerprint") : nothing), EP_ARTIFACT, "r")
    if fp_ep !== nothing && length(μ_ep) == K && maximum(abs.(fp_ep .- v_fingerprint(graph.V))) <= 1e-9
        t_ep = @elapsed begin
            global w_map_ep, map_ep_iters, _ = find_map(μ_ep, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA;
                max_iters = TOTAL_MAP_ITERS ÷ length(S_GRID), reaction = REACTION)
        end
        global map_time_seconds += t_ep
        println(@sprintf("[nu=%s rho=%s idx=%d] MAP start (ii), Adam from the EP mean: %d iters in %.1fs, nlp %.6e",
            NU, RHO, TRAJ_IDX, map_ep_iters, t_ep, nlp_and_grad(w_map_ep)[1]))
    else
        println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] MAP start (ii) skipped: EP artifact has no matching eigenbasis fingerprint")
    end
else
    println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] MAP start (ii) skipped: no EP artifact at $EP_ARTIFACT")
end

"""
    gauss_newton_full(w) -> Symmetric matrix

Exact Gauss-Newton curvature of the target at `w`: `J_icᵀJ_ic/β_ic² + J_rᵀJ_r/β_pde² + I/σw²`
with `J_ic = g'(z_ic)⊙Φ_ic` and, for the Fisher residual `R = g'(z)·Q − ρ·g(z)(1−g(z))`,
`J_r = (g''(z)·Q − ρ·(1−2g(z))·g'(z))⊙Φ_r + g'(z)⊙R_mat`. Always SPD; the dense mass matrix.
"""
function gauss_newton_full(w)
    z_ic = Φ_ic * w
    gp_ic = piecewise_activation_derivative.(z_ic, PW_ALPHA)
    z_r = Φ_r * w
    Q_r = R_mat * w
    g_r = piecewise_activation.(z_r, PW_ALPHA)
    gp_r = piecewise_activation_derivative.(z_r, PW_ALPHA)
    gpp_r = GDPINN.piecewise_activation_second_derivative.(z_r, PW_ALPHA)
    J_ic = gp_ic .* Φ_ic
    J_r = (gpp_r .* Q_r .- REACTION .* (1 .- 2 .* g_r) .* gp_r) .* Φ_r .+ gp_r .* R_mat
    H = (J_ic' * J_ic) ./ BETA_IC^2 .+ (J_r' * J_r) ./ BETA_PDE^2
    H[diagind(H)] .+= 1.0 / SIGMA_W^2
    return Symmetric(H)
end

# Gauss-Newton polish of both candidates; the lower negative log posterior wins.
candidates = [("zero-ladder", w_map_zero)]
w_map_ep === nothing || push!(candidates, ("ep-mean", w_map_ep))
polished = []
gn_time_seconds = @elapsed for (name, w_c) in candidates
    nlp_c = nlp_and_grad(w_c)[1]
    w_p, it_p, hist_p = refine_map_gauss_newton(w_c, nlp_and_grad, gauss_newton_full)
    push!(polished, (name = name, w = w_p, iters = it_p, nlp_before = nlp_c, nlp = hist_p[end]))
    println(@sprintf("[nu=%s rho=%s idx=%d] MAP Gauss-Newton polish (%s): %d steps, nlp %.6e -> %.6e",
        NU, RHO, TRAJ_IDX, name, it_p, nlp_c, hist_p[end]))
end
map_time_seconds += gn_time_seconds
best = polished[argmin([c.nlp for c in polished])]
w_map, map_gn_iters, nlp_adam = best.w, best.iters, best.nlp_before
map_gn_history = [best.nlp]
map_init = best.name
map_nlp_by_start = Dict(c.name => c.nlp for c in polished)
println(@sprintf("[nu=%s rho=%s idx=%d] MAP: start '%s' wins with nlp %.6e (%s)", NU, RHO, TRAJ_IDX, map_init, best.nlp,
    join(["$(c.name)=$(round(c.nlp, sigdigits = 7))" for c in polished], ", ")))

# Branch check of the warm start: the Fisher posterior has a trivial u ≡ 0 branch;
# the exact objective's data term makes it a poor local optimum, and Adam + Gauss-Newton from
# w = 0 lands on the data branch (rel-L2 of the MAP fit against the stored trajectory well below
# 0.5). Recorded in the artifact and flagged in the log if it ever fails.
map_rel_l2 = metrics(piecewise_activation.(predict_grid(p, graph.V * w_map), PW_ALPHA), u).rel_l2
map_on_data_branch = map_rel_l2 < 0.5
println(@sprintf("[nu=%s rho=%s idx=%d] MAP fit: rel-L2 vs stored trajectory %.4f -> %s", NU, RHO, TRAJ_IDX, map_rel_l2,
    map_on_data_branch ? "data branch" : "WARNING: trivial branch (u ~ 0), the chains start on the wrong mode"))

H_gn = gauss_newton_full(w_map)
mass_chol = cholesky(H_gn)

sample_momentum(rng) = hmc_momentum(rng, mass_chol)
kinetic(pvec) = hmc_kinetic(pvec, mass_chol)
jitter_direction(rng) = mass_chol.U \ randn(rng, K)

function hmc_chain(w_init, chain_id, ε0)
    rng_c = MersenneTwister(100 + chain_id)
    w = copy(w_init)
    nlp_w, _ = nlp_and_grad(w)
    kept = Matrix{Float64}(undef, K, NUM_SAMPLES - BURN)
    n_accept = 0
    n_accept_kept = 0
    n_kept = 0
    da = DualAveraging(ε0; δ = TARGET_ACCEPT)
    ε = ε0
    ε_trace = Float64[]
    for k in 1:NUM_SAMPLES
        p0 = sample_momentum(rng_c)
        H0 = nlp_w + kinetic(p0)
        L_k = rand(rng_c, 1:LEAPFROG_L)   # randomized trajectory length (Neal 2011, Sec. 3.2), see the header
        w_prop, p_prop = leapfrog_mass(w, p0, ε, L_k, nlp_and_grad, mass_chol)
        nlp_prop, _ = nlp_and_grad(w_prop)
        α = BPINN.accept_prob(H0, nlp_prop + kinetic(p_prop))
        if rand(rng_c) < α
            w = w_prop
            nlp_w = nlp_prop
            n_accept += 1
            k > BURN && (n_accept_kept += 1)
        end
        if k <= BURN
            push!(ε_trace, ε)
            ε = adapt_step!(da, α)
            k == BURN && (ε = adapted_step_size(da))
        end
        k > BURN && (kept[:, n_kept += 1] = w)
    end
    return kept, n_accept / NUM_SAMPLES, n_accept_kept / (NUM_SAMPLES - BURN), ε, ε_trace
end

w_inits = Vector{Vector{Float64}}(undef, N_CHAINS)
chain_init_step_sizes = Vector{Float64}(undef, N_CHAINS)
rng_init = MersenneTwister(42)
for c in 1:N_CHAINS
    w_inits[c] = c == 1 ? copy(w_map) : w_map .+ JITTER .* jitter_direction(rng_init)
    chain_init_step_sizes[c] = hg_initial_step_size(w_inits[c], nlp_and_grad, mass_chol;
        L = LEAPFROG_L, rng = MersenneTwister(c == 1 ? 1 : 200 + c))
end
step_size = chain_init_step_sizes[1]
println(@sprintf("[nu=%s rho=%s idx=%d] initial step sizes (Hoffman-Gelman search per chain): %s",
    NU, RHO, TRAJ_IDX, string(chain_init_step_sizes)))

chains = Vector{Matrix{Float64}}(undef, N_CHAINS)
accept_rates = Vector{Float64}(undef, N_CHAINS)
accept_rates_kept = Vector{Float64}(undef, N_CHAINS)
chain_times = Vector{Float64}(undef, N_CHAINS)
chain_step_sizes = Vector{Float64}(undef, N_CHAINS)
step_traces = Vector{Vector{Float64}}(undef, N_CHAINS)
for c in 1:N_CHAINS
    chain_times[c] = @elapsed (chains[c], accept_rates[c], accept_rates_kept[c], chain_step_sizes[c], step_traces[c]) =
        hmc_chain(w_inits[c], c, chain_init_step_sizes[c])
    println(@sprintf("[nu=%s rho=%s idx=%d] chain %d: step %.3e -> %.3e (adapted over %d warm-up iterations) accept=%.3f (retained draws %.3f) in %.1fs",
        NU, RHO, TRAJ_IDX, c, chain_init_step_sizes[c], chain_step_sizes[c], BURN, accept_rates[c],
        accept_rates_kept[c], chain_times[c]))
    flush(stdout)
end
train_time_seconds = map_time_seconds + sum(chain_times)

all_samples = [chains[c][:, j] for c in 1:N_CHAINS for j in 1:size(chains[c], 2)]
predict_time_seconds = @elapsed begin
    w_mean = sum(all_samples) ./ length(all_samples)
    w = graph.V * w_mean
    samples_mat = reduce(hcat, all_samples)
    w_eigen_mean = w_mean
    w_eigen_std = vec(std(samples_mat, dims = 2))
    pred_u, pred_var = predict_bpinn_stats(p, graph.V, all_samples; α = PW_ALPHA)
end
residual = predict_reacdiff_residual(p, w, NU_NORM, RHO_NORM; α = PW_ALPHA)  # at the posterior-mean weights
m = metrics(pred_u, u)
println(@sprintf("[nu=%s rho=%s idx=%d] pooled %d samples, predicted in %.2fs | %s | mean|residual|=%.3e",
    NU, RHO, TRAJ_IDX, length(all_samples), predict_time_seconds, fmt(m), Statistics.mean(abs.(residual))))

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
    attributes(f)["map_time_seconds"] = map_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["num_samples"] = NUM_SAMPLES
    attributes(f)["burn"] = BURN
    attributes(f)["leapfrog_L"] = LEAPFROG_L                  # the MAXIMUM number of leapfrog steps per proposal
    attributes(f)["trajectory_length_rule"] = "L ~ Uniform{1..leapfrog_L} per proposal (Neal 2011); warm-up dual averaging of the step size to target 0.8"
    attributes(f)["n_chains"] = N_CHAINS
    attributes(f)["jitter"] = JITTER
    attributes(f)["step_size"] = step_size
    attributes(f)["target_accept"] = TARGET_ACCEPT
    attributes(f)["step_size_protocol"] = "HG-Alg4 init per chain + dual averaging over burn-in to target 0.8, fixed afterwards"
    attributes(f)["map_gn_iters"] = map_gn_iters
    attributes(f)["map_adam_iters"] = map_iters
    attributes(f)["map_rung_iters"] = map_rung_iters
    attributes(f)["map_init"] = map_init
    attributes(f)["map_nlp_zero_ladder"] = map_nlp_by_start["zero-ladder"]
    attributes(f)["map_nlp_ep_mean"] = get(map_nlp_by_start, "ep-mean", NaN)
    attributes(f)["map_ep_adam_iters"] = map_ep_iters
    attributes(f)["map_nlp_adam"] = nlp_adam
    attributes(f)["map_nlp"] = map_gn_history[end]
    attributes(f)["map_rel_l2"] = map_rel_l2
    attributes(f)["map_on_data_branch"] = Int(map_on_data_branch)
    attributes(f)["rel_l2"] = m.rel_l2
    write(f, "accept_rates", accept_rates)
    write(f, "accept_rates_kept", accept_rates_kept)
    write(f, "chain_times", chain_times)
    write(f, "chain_step_sizes", chain_step_sizes)
    write(f, "chain_init_step_sizes", chain_init_step_sizes)
    for c in 1:N_CHAINS
        write(f, "step_trace_chain$(c)", step_traces[c])
    end
    write(f, "mass_diag", diag(H_gn))
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)
    write(f, "w_eigen_mean", w_eigen_mean)
    write(f, "w_eigen_std", w_eigen_std)
    write(f, "v_fingerprint", v_fingerprint(graph.V))
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "pred_var", pred_var)
    write(f, "residual", residual)
    for c in 1:N_CHAINS
        write(f, "samples_chain$(c)", chains[c])  # K x (num_samples-burn)
    end
end
println("[nu=$NU rho=$RHO idx=$TRAJ_IDX] saved to $OUT_PATH")
