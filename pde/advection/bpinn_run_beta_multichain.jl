# Multichain HMC reference for ONE advection wave speed: same architecture, basis,
# eigenbasis, and data as `mp_pinn_run_beta.jl`, with the B-PINN target density of `bpinn_model.jl`
# (Yang et al. 2021's exact posterior), sampled properly enough to serve as a gold standard:
#
#   * `n_chains` independent HMC chains (chain 1 starts at the MAP, the others at
#     overdispersed jittered points), each with its own momentum rng -- so split-R-hat and
#     per-coordinate ESS can certify convergence instead of assuming it;
#   * a DENSE mass matrix: the exact Gauss-Newton Hessian of the target at the MAP
#     (M = J_icᵀJ_ic/β_ic² + J_rᵀJ_r/β_pde² + I/σw², Cholesky-factorized once). The
#     posterior's per-coordinate scales span ~3 orders of magnitude in the eigenbasis AND
#     retain strong cross-coordinate correlations from the activation nonlinearity; with a
#     unit (or even diagonal) mass, the stable step size is set by the tightest direction and
#     the rest mix glacially (measured: unit mass ~ ESS 1-10; diagonal GN mass ~ median ESS
#     47 of 3000 draws, only 35% of coordinates passing R-hat < 1.05). The dense GN metric
#     whitens the near-Gaussian target, giving near-independent draws per proposal.
#     Preconditioning changes only the mixing speed, never the stationary distribution, so
#     this still samples the exact posterior -- the diagnostics verify it.
#
#   * MAP warm start = Adam (`find_map`) followed by a damped Gauss-Newton polish
#     (`refine_map_gauss_newton`; Adam alone stalls short of the mode at large K), and the mass
#     matrix is evaluated at the polished MAP;
#   * step size: per chain, Hoffman & Gelman's one-draw search from the
#     chain's own initial point gives ε0, dual averaging over the BURN warm-up iterations adapts
#     it towards the acceptance target 0.8 (`DualAveraging`), and the retained draws use the
#     averaged step size, fixed. A step size fixed from the start freezes the chains of
#     b=1 idx 6451 and b=7 idx 5906/898 (acceptance 0.00);
#   * the number of leapfrog steps of every proposal is drawn uniformly from 1..L (L = 20 by
#     default, the maximum): with the step adapted to the integrator's stability
#     limit, a FIXED trajectory of 20 steps is close to one full period of the whitened target's
#     oscillation, so every proposal returns near its start (lag-1 autocorrelation +0.99 at b=2
#     with fixed L = 20, against -0.75 with a smaller fixed step); randomizing the
#     trajectory length is the standard remedy for this near-periodicity (Neal 2011, Sec. 3.2).
#
# Full per-chain sample matrices are saved so the figure scripts can compute R-hat/ESS and
# any posterior functional without re-running. Every advection reference artifact
# (`artifacts/bpinn_mc_beta*_idx*.h5`) is produced under this protocol.
#
# Usage: julia --project=. pde/advection/bpinn_run_beta_multichain.jl <wave_speed> [traj_idx] [out_path]
#            [num_samples] [burn] [L] [n_chains] [jitter]

include(joinpath(@__DIR__, "..", "bpinn_model.jl"))

using .BPINN
using .BPINN.GDPINN
using .BPINN.GDPINN.MPPINN
using .BPINN.GDPINN.MPPINN.Factors: piecewise_activation_derivative
using Random
using Printf
using Statistics
using LinearAlgebra
using HDF5

WAVE_SPEED = parse(Float64, ARGS[1])
TRAJ_IDX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : -1
NUM_SAMPLES = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
BURN = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 250
# Under the dense GN metric the whitened target is ~unit-scale, so short trajectories already
# decorrelate; L=20 keeps a safety margin at moderate cost per proposal.
LEAPFROG_L = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20
N_CHAINS = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 4
# Overdispersion of the non-MAP chains' initial points, in units of the Gauss-Newton posterior
# scale: init = MAP + JITTER·ξ with ξ ~ N(0, M⁻¹) (posterior-shaped). Absolute isotropic jitter
# fails badly: the posterior scales span ~3 orders of magnitude across coordinates, so a fixed
# absolute jitter is hundreds of σ in the tight ones -- leapfrog's energy error scales with the
# oscillation amplitude in σ units, and such a chain rejects every proposal (observed: 0.001).
# JITTER itself must stay moderate: per-coordinate jitter j means a WHITENED displacement of
# j·sqrt(K); at K=577, j=5 puts chains where the local curvature no longer matches the MAP-based
# metric and leapfrog diverges at a shared step size (observed: accept 0.0, R-hat 7.5).
# j=2 plus a per-chain step-size search at each chain's own init (below) avoids this.
JITTER = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 2.0

ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(ARTIFACTS_DIR, "bpinn_mc_beta$(WAVE_SPEED)_idx$(TRAJ_IDX).h5")

# Same hyperparameters and construction as mp_pinn_run_beta.jl.
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
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX] c_norm=$c_norm -> Kt=$Kt " *
        "(multichain HMC: $N_CHAINS chains x $NUM_SAMPLES samples, burn=$BURN, L=$LEAPFROG_L)")

basis = TrigBasis(Kx, Kt)
p = Problem(basis, u, x, t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
rng = MersenneTwister(7)  # identical to every other runner -> same V/K/data
graph = build_graph(p, rng; n_ic = N_IC, nt = Kt, nx = NC_X, σw = SIGMA_W, β_ic = BETA_IC,
    β_pde = BETA_PDE, α = PW_ALPHA)
Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde = design_matrices(graph)
K = size(graph.V, 2)
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX] K=$(dim(basis)) raw -> $K kept")

nlp_and_grad(w) = neg_log_posterior_and_grad(w, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA)

w0 = zeros(K)
map_time_seconds = @elapsed begin
    w_map, map_iters, _ = find_map(w0, Φ_ic, y_ic, β_ic, Φ_r, R_mat, β_pde, SIGMA_W, PW_ALPHA)
end
println(@sprintf("[beta=%s idx=%d] MAP warm start: %d iters in %.1fs", WAVE_SPEED, TRAJ_IDX, map_iters, map_time_seconds))

"""
    gauss_newton_full(w) -> Symmetric matrix

Exact Gauss-Newton curvature of `neg_log_posterior_and_grad`'s target at `w`:
`J_icᵀJ_ic/β_ic² + J_rᵀJ_r/β_pde² + I/σw²`, with `J_ic = g'(z_ic)⊙Φ_ic` (data-residual
Jacobian) and `J_r = g''(z_r)⊙Q⊙Φ_r + g'(z_r)⊙R_mat` (PDE-residual Jacobian) -- always SPD.
Used as the dense mass matrix.
"""
function gauss_newton_full(w)
    z_ic = Φ_ic * w
    gp_ic = piecewise_activation_derivative.(z_ic, PW_ALPHA)
    z_r = Φ_r * w
    Q_r = R_mat * w
    gp_r = piecewise_activation_derivative.(z_r, PW_ALPHA)
    gpp_r = GDPINN.piecewise_activation_second_derivative.(z_r, PW_ALPHA)
    J_ic = gp_ic .* Φ_ic
    J_r = gpp_r .* Q_r .* Φ_r .+ gp_r .* R_mat
    H = (J_ic' * J_ic) ./ BETA_IC^2 .+ (J_r' * J_r) ./ BETA_PDE^2
    H[diagind(H)] .+= 1.0 / SIGMA_W^2
    return Symmetric(H)
end

# Damped Gauss-Newton polish of the Adam warm start (`BPINN.refine_map_gauss_newton`): Adam
# stalls short of the mode on the large-K problems (b=7: nlp still oscillating by 10-30 % at the
# 30k-iteration cap), which would leave the chains' start and the metric off the mode.
nlp_adam = nlp_and_grad(w_map)[1]
gn_time_seconds = @elapsed begin
    w_map, map_gn_iters, map_gn_history = refine_map_gauss_newton(w_map, nlp_and_grad, gauss_newton_full)
end
map_time_seconds += gn_time_seconds
println(@sprintf("[beta=%s idx=%d] MAP Gauss-Newton polish: %d steps in %.1fs, nlp %.6e -> %.6e",
    WAVE_SPEED, TRAJ_IDX, map_gn_iters, gn_time_seconds, nlp_adam, map_gn_history[end]))

H_gn = gauss_newton_full(w_map)
mass_chol = cholesky(H_gn)  # momentum p ~ N(0, M), M = L·Lᵀ

sample_momentum(rng) = hmc_momentum(rng, mass_chol)            # p ~ N(0, M)
kinetic(pvec) = hmc_kinetic(pvec, mass_chol)                   # ½ pᵀ M⁻¹ p
jitter_direction(rng) = mass_chol.U \ randn(rng, K)  # N(0, M⁻¹) draw: posterior-shaped jitter

# Step-size protocol: per chain, Hoffman & Gelman's
# one-draw doubling/halving search from the chain's own initial point gives ε0; during the BURN
# warm-up iterations ε is adapted by dual averaging towards the acceptance target TARGET_ACCEPT
# (`BPINN.DualAveraging`); the retained draws then use the averaged step size, fixed. A step
# chosen once at the start freezes chains whose posterior is more curved
# away from the mode than the MAP-based metric predicts (b=1 idx 6451, b=7 idx 5906/898).
TARGET_ACCEPT = 0.8

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
            k == BURN && (ε = adapted_step_size(da))   # fixed from here on
        end
        k > BURN && (kept[:, n_kept += 1] = w)
    end
    return kept, n_accept / NUM_SAMPLES, n_accept_kept / (NUM_SAMPLES - BURN), ε, ε_trace
end

# Initial points and initial step sizes (the jitter and search generators are separate from the
# chains' generators).
w_inits = Vector{Vector{Float64}}(undef, N_CHAINS)
chain_init_step_sizes = Vector{Float64}(undef, N_CHAINS)
rng_init = MersenneTwister(42)
for c in 1:N_CHAINS
    w_inits[c] = c == 1 ? copy(w_map) : w_map .+ JITTER .* jitter_direction(rng_init)
    chain_init_step_sizes[c] = hg_initial_step_size(w_inits[c], nlp_and_grad, mass_chol;
        L = LEAPFROG_L, rng = MersenneTwister(c == 1 ? 1 : 200 + c))
end
step_size = chain_init_step_sizes[1]
println(@sprintf("[beta=%s idx=%d] initial step sizes (Hoffman-Gelman search per chain): %s",
    WAVE_SPEED, TRAJ_IDX, string(chain_init_step_sizes)))

chains = Vector{Matrix{Float64}}(undef, N_CHAINS)
accept_rates = Vector{Float64}(undef, N_CHAINS)
accept_rates_kept = Vector{Float64}(undef, N_CHAINS)
chain_times = Vector{Float64}(undef, N_CHAINS)
chain_step_sizes = Vector{Float64}(undef, N_CHAINS)
step_traces = Vector{Vector{Float64}}(undef, N_CHAINS)
for c in 1:N_CHAINS
    chain_times[c] = @elapsed (chains[c], accept_rates[c], accept_rates_kept[c], chain_step_sizes[c], step_traces[c]) =
        hmc_chain(w_inits[c], c, chain_init_step_sizes[c])
    println(@sprintf("[beta=%s idx=%d] chain %d: step %.3e -> %.3e (adapted over %d warm-up iterations) accept=%.3f (retained draws %.3f) in %.1fs",
        WAVE_SPEED, TRAJ_IDX, c, chain_init_step_sizes[c], chain_step_sizes[c], BURN, accept_rates[c],
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
println(@sprintf("[beta=%s idx=%d] pooled %d samples, predicted in %.2fs",
    WAVE_SPEED, TRAJ_IDX, length(all_samples), predict_time_seconds))

h5open(OUT_PATH, "w") do f
    attributes(f)["beta"] = WAVE_SPEED
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["train_time_seconds"] = train_time_seconds
    attributes(f)["map_time_seconds"] = map_time_seconds
    attributes(f)["predict_time_seconds"] = predict_time_seconds
    attributes(f)["Kx"] = basis.Kx
    attributes(f)["Kt"] = basis.Kt
    attributes(f)["c_norm"] = c_norm
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = BPINN.GDPINN.Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    attributes(f)["num_samples"] = NUM_SAMPLES
    attributes(f)["burn"] = BURN
    attributes(f)["leapfrog_L"] = LEAPFROG_L                  # the MAXIMUM number of leapfrog steps per proposal
    attributes(f)["trajectory_length_rule"] = "L ~ Uniform{1..leapfrog_L} per proposal (Neal 2011); warm-up dual averaging of the step size to target 0.8"
    attributes(f)["n_chains"] = N_CHAINS
    attributes(f)["jitter"] = JITTER
    attributes(f)["step_size"] = step_size                      # chain 1's initial step (Hoffman-Gelman search at the MAP)
    attributes(f)["target_accept"] = TARGET_ACCEPT
    attributes(f)["step_size_protocol"] = "HG-Alg4 init per chain + dual averaging over burn-in to target 0.8, fixed afterwards"
    attributes(f)["map_gn_iters"] = map_gn_iters
    attributes(f)["map_nlp_adam"] = nlp_adam
    attributes(f)["map_nlp"] = map_gn_history[end]
    write(f, "accept_rates", accept_rates)                      # over all NUM_SAMPLES proposals (warm-up included)
    write(f, "accept_rates_kept", accept_rates_kept)            # over the retained draws only
    write(f, "chain_times", chain_times)
    write(f, "chain_step_sizes", chain_step_sizes)              # the adapted, fixed step of each chain
    write(f, "chain_init_step_sizes", chain_init_step_sizes)
    for c in 1:N_CHAINS
        write(f, "step_trace_chain$(c)", step_traces[c])        # warm-up step sizes, one per warm-up iteration
    end
    write(f, "mass_diag", diag(H_gn))
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "w", w)
    write(f, "w_eigen_mean", w_eigen_mean)
    write(f, "w_eigen_std", w_eigen_std)
    write(f, "true_u", u)
    write(f, "pred_u", pred_u)
    write(f, "pred_var", pred_var)
    for c in 1:N_CHAINS
        write(f, "samples_chain$(c)", chains[c])  # K x (num_samples-burn)
    end
end
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX] saved to $OUT_PATH")
