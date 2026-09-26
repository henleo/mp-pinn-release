# Builds IDENTICAL factor graphs (same eigenbasis V, same collocation points, same prior --
# guaranteed by re-running build_graph with a fresh but identically-seeded rng each time) and
# trains each on a different factor subset: IC/data factors only, PDE-residual factors only, or
# both (the normal full training). Also captures the untrained (prior) state before any training.
# For each regime, records the FULL (x,t) grid of u's predictive mean/std from the closed-form
# moments of the one-pass runner (`predict_moments`) -- the same numbers the
# per-point `predict_slice` gives (to machine precision), without its per-point memory growth
# (~6 GiB at b=7), so mean and std come from one consistent Gaussian belief per (x,t) point.
#
# Purpose: isolate how much of the trained prediction envelope's narrowing (prior -> posterior)
# is due to the IC/data subgraph vs the PDE-residual subgraph, since they are literally two
# disjoint factor lists (`data_factors`, `res_factors`) updating the same `db_marginal` -- running
# each in isolation is a clean ablation, not an approximation.
#
# Usage: julia --project=. pde/advection/mp_pinn_subgraph_ablation.jl [wave_speed] [traj_idx] [out_path] [regimes]
#
# `regimes` is a comma-separated subset of prior,ic,res,full (default: all four). The full regime
# is exactly `train!` (same factor order, same rng), so it reproduces the one-pass artifact
# `mp_pinn_beta{b}_idx{i}.h5` of `mp_pinn_run_beta.jl` bit for bit; `mp_pinn_subgraph_ablation_sweep.py`
# therefore runs prior,ic,res only and reads the full regime from that artifact.
#
# Per regime r the file holds mean_r / std_r (n_x x n_t), the eigenbasis posterior
# w_eigen_mean_r / w_eigen_std_r, the raw mean weights w_r = V*mu, the max-KL history kl_r and
# the attributes epochs_r / train_time_seconds_r (0 epochs for the prior).

include(joinpath(@__DIR__, "..", "mp_pinn_model.jl"))

using .MPPINN
using .MPPINN.Factors
using Random
using Printf
using HDF5

WAVE_SPEED = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
TRAJ_IDX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6369
ARTIFACTS_DIR = joinpath(@__DIR__, "..", "..", "artifacts")
mkpath(ARTIFACTS_DIR)
OUT_PATH = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(ARTIFACTS_DIR, "subgraph_ablation_beta$(WAVE_SPEED)_idx$(TRAJ_IDX).h5")
REGIMES = length(ARGS) >= 4 ? String.(split(ARGS[4], ",")) : ["prior", "ic", "res", "full"]
all(r -> r in ("prior", "ic", "res", "full"), REGIMES) ||
    error("regimes must be a comma-separated subset of prior,ic,res,full; got $(ARGS[4])")

TIME_FILL = 0.5
Kx = 5
RESONANCE_MARGIN = 3
N_IC = 1024
NC_X = 48
BETA_IC = 0.02
BETA_PDE = 0.05
SIGMA_W = 1.0
PW_ALPHA = 0.9

Factors.MSG_DEBUG[] = 0

u, x, t = load_slice(; wave_speed = WAVE_SPEED, idx = TRAJ_IDX)
c_norm = estimate_c_norm(t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)
Kt = ceil(Int, Kx * abs(c_norm)) + RESONANCE_MARGIN
println("[beta=$WAVE_SPEED idx=$TRAJ_IDX] c_norm=$c_norm -> Kt=$Kt, regimes $(join(REGIMES, ","))")

basis = TrigBasis(Kx, Kt)
p = Problem(basis, u, x, t; wave_speed = WAVE_SPEED, time_fill = TIME_FILL)

fresh_graph() = build_graph(p, MersenneTwister(7); n_ic = N_IC, nt = Kt, nx = NC_X,
    σw = SIGMA_W, β_ic = BETA_IC, β_pde = BETA_PDE, α = PW_ALPHA)

# Mirrors `train!` exactly, except `order` is restricted to the requested factor subset(s).
# Directions with zero precision contribution from the active subset simply never move off the
# prior -- not an error, that IS the ablation result.
function train_subset!(graph::NLFactorGraph; use_data::Bool, use_residual::Bool,
                        max_epochs = 300, kl_tol = 1e-5, check_every = 20,
                        rng = MersenneTwister(11))
    n_data = length(graph.data_factors)
    n_res = length(graph.res_factors)
    idxs = Int[]
    use_data && append!(idxs, 1:n_data)
    use_residual && append!(idxs, (n_data + 1):(n_data + n_res))
    isempty(idxs) && return 0, Float64[]
    kl_history = Float64[]
    epochs_done = 0
    for epoch in 1:max_epochs
        maxkl = 0.0
        shuffle!(rng, idxs)
        for idx in idxs
            kl = if idx <= n_data
                f = graph.data_factors[idx]
                forward = Factors.data_forward_message(f)
                Factors.data_backward_update!(f, forward)
            else
                f = graph.res_factors[idx - n_data]
                forward = Factors.residual_forward_message(f)
                Factors.residual_backward_update!(f, forward)
            end
            maxkl = max(maxkl, kl)
        end
        push!(kl_history, maxkl)
        epochs_done = epoch
        if epoch % check_every == 0 || maxkl < kl_tol
            @printf("    epoch %3d  max KL %.3e\n", epoch, maxkl)
            flush(stdout)
        end
        maxkl < kl_tol && break
    end
    return epochs_done, kl_history
end

# Predictive grids and posterior of one regime's graph, in the one-pass runner's closed form.
function snapshot(graph::NLFactorGraph, epochs, kl_history, train_time)
    post = posterior_parameters(graph)
    mean_grid, var_grid = predict_moments(graph, p; α = PW_ALPHA)
    return (mean = mean_grid, std = sqrt.(var_grid), μ = post.μ_vec, σ = post.σ_vec,
            w = graph.V * post.μ_vec, epochs = epochs, kl = kl_history, train_time = train_time)
end

results = Dict{String, Any}()
K_kept = 0
if "prior" in REGIMES || "ic" in REGIMES
    println("=== prior (untrained)", "ic" in REGIMES ? " + IC-only ===" : " ===")
    g = fresh_graph()
    global K_kept = size(g.V, 2)
    "prior" in REGIMES && (results["prior"] = snapshot(g, 0, Float64[], 0.0))
    if "ic" in REGIMES
        tt = @elapsed (ep, kl) = train_subset!(g; use_data = true, use_residual = false)
        results["ic"] = snapshot(g, ep, kl, tt)
    end
    g = nothing; GC.gc()
end
if "res" in REGIMES
    println("=== residual-only ===")
    g = fresh_graph()
    global K_kept = size(g.V, 2)
    tt = @elapsed (ep, kl) = train_subset!(g; use_data = false, use_residual = true)
    results["res"] = snapshot(g, ep, kl, tt)
    g = nothing; GC.gc()
end
if "full" in REGIMES
    println("=== full (IC + residual) ===")
    g = fresh_graph()
    global K_kept = size(g.V, 2)
    tt = @elapsed (ep, kl) = train_subset!(g; use_data = true, use_residual = true)
    results["full"] = snapshot(g, ep, kl, tt)
    g = nothing; GC.gc()
end

h5open(OUT_PATH, "w") do f
    attributes(f)["beta"] = WAVE_SPEED
    attributes(f)["idx"] = TRAJ_IDX
    attributes(f)["regimes"] = join(REGIMES, ",")
    attributes(f)["Kx"] = Kx
    attributes(f)["Kt"] = Kt
    attributes(f)["K_kept"] = K_kept
    attributes(f)["c_norm"] = c_norm
    attributes(f)["alpha"] = PW_ALPHA
    attributes(f)["piecewise_beta"] = Factors.PIECEWISE_BETA
    attributes(f)["sigma_w"] = SIGMA_W
    attributes(f)["beta_ic"] = BETA_IC
    attributes(f)["beta_pde"] = BETA_PDE
    write(f, "x", x)
    write(f, "t", t)
    write(f, "tn", p.tn)
    write(f, "true_u", u)
    for r in REGIMES
        s = results[r]
        write(f, "mean_$r", s.mean); write(f, "std_$r", s.std)
        write(f, "w_eigen_mean_$r", s.μ); write(f, "w_eigen_std_$r", s.σ)
        write(f, "w_$r", s.w)
        isempty(s.kl) || write(f, "kl_$r", s.kl)   # the prior has no training history
        attributes(f)["epochs_$r"] = s.epochs
        attributes(f)["train_time_seconds_$r"] = s.train_time
    end
end
println("saved to $OUT_PATH")
