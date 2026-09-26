# Steady-state (compiled) inference re-timing, robust protocol (`run_experiments.py` keeps its HMC
# rows; the closed-form rows are re-timed with more repetitions by `warm_retime_closed_form_v3.jl`).
#   * all families (onepass, vi, hmc) in ONE process,
#     INTERLEAVED per (beta, idx) so Ours and VI are measured back-to-back under the same machine state;
#   * min-of-K repetitions per artifact (K=3) after one untimed warm-up per code path, GC flushed
#     before each artifact -- the standard estimator of steady-state compute cost;
#   * restricted to the timing figure's betas and the trajectories common to Ours/HMC/VI;
#   * every re-timed path must reproduce the artifact's saved pred_var (max |Δ| ~ 0).
# Usage: julia --project=. pde/advection/diagnostics/steady_state_timing/warm_retime_hmc_v2.jl <out.csv> [beta]
include(joinpath(@__DIR__, "..", "..", "..", "vi_model.jl"))
using .VIPINN
using .VIPINN.BPINN
using .VIPINN.BPINN.GDPINN.MPPINN
using Random, HDF5, Printf, Statistics, LinearAlgebra

REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
OUT = ARGS[1]
KREP = 3
PW_ALPHA = 0.9
# Optional second argument: a single wave speed to re-time (e.g. 7.0); default = all five.
BETAS = length(ARGS) >= 2 ? [parse(Float64, ARGS[2])] : [0.1, 0.4, 1.0, 2.0, 7.0]
IDXS = [9394, 898, 2398, 5906, 2343, 8225, 5506, 6451, 2670, 3497]
PREFIX = Dict("onepass" => "mp_pinn_beta", "vi" => "vi_beta", "hmc" => "bpinn_mc_beta")
FAMS = ["onepass", "vi", "hmc"]
art_path(fam, b, i) = joinpath(REPO, "artifacts", "$(PREFIX[fam])$(b)_idx$(i).h5")

cache = Dict{Float64,Any}()
function problem_and_V(beta::Float64, idx::Int, Kt_attr::Int)
    haskey(cache, beta) && return cache[beta]
    u, x, t = load_slice(; wave_speed = beta, idx = idx)
    c = estimate_c_norm(t; wave_speed = beta, time_fill = 0.5)
    Kt = ceil(Int, 5 * abs(c)) + 3
    Kt == Kt_attr || error("Kt mismatch at beta=$beta")
    p = Problem(TrigBasis(5, Kt), u, x, t; wave_speed = beta, time_fill = 0.5)
    g = build_graph(p, MersenneTwister(7); n_ic = 1024, nt = Kt, nx = 48, σw = 1.0, β_ic = 0.02, β_pde = 0.05, α = PW_ALPHA)
    cache[beta] = (p, g.V)
    return cache[beta]
end

function timed_block(fam, p, V, art)
    if fam == "onepass" || fam == "vi"
        w = V * art.μ
        return predict_moments_grid(V, p, art.μ, art.σ .^ 2; α = PW_ALPHA)
    else
        samples = art.samples
        w_mean = sum(samples) ./ length(samples)
        w = V * w_mean
        samples_mat = reduce(hcat, samples)
        w_eigen_std = vec(std(samples_mat, dims = 2))
        return predict_bpinn_stats(p, V, samples; α = PW_ALPHA)
    end
end

function load_art(fam, f)
    h5open(f, "r") do h
        Kt = Int(read(attributes(h)["Kt"])); cold = Float64(read(attributes(h)["predict_time_seconds"]))
        pv = Float64.(read(h, "pred_var"))
        if fam == "hmc"
            nch = Int(read(attributes(h)["n_chains"])); samples = Vector{Vector{Float64}}()
            for c in 1:nch
                M = Float64.(read(h, "samples_chain$c")); append!(samples, [M[:, j] for j in 1:size(M, 2)])
            end
            return (Kt = Kt, cold = cold, pv = pv, samples = samples, K = length(samples[1]))
        else
            μ = Float64.(read(h, "w_eigen_mean"))
            return (Kt = Kt, cold = cold, pv = pv, μ = μ, σ = Float64.(read(h, "w_eigen_std")), K = length(μ))
        end
    end
end

open(OUT, "w") do io
    println(io, "artifact,family,beta,idx,K,warm_predict_seconds,warm_median3_seconds,cold_predict_seconds,pred_var_maxdiff")
end
warmed = Set{String}()   # code paths already compiled: "closed" (onepass/vi) and "hmc"
for beta in BETAS
    common = [i for i in IDXS if all(isfile(art_path(f, beta, i)) for f in ("onepass", "vi", "hmc"))]
    println("beta=$beta common idxs: $common")
    for idx in common, fam in FAMS
        f = art_path(fam, beta, idx); isfile(f) || continue
        art = load_art(fam, f)
        p, V = problem_and_V(beta, idx, art.Kt)
        path = fam == "hmc" ? "hmc" : "closed"
        if !(path in warmed); timed_block(fam, p, V, art); push!(warmed, path); end
        GC.gc()
        ts = Float64[]; local pu, pv
        for r in 1:KREP
            t = @elapsed (pu, pv) = timed_block(fam, p, V, art); push!(ts, t)
        end
        dv = maximum(abs.(pv .- art.pv))
        open(OUT, "a") do io
            @printf(io, "%s,%s,%g,%d,%d,%.4f,%.4f,%.4f,%.3e\n", basename(f), fam, beta, idx, art.K, minimum(ts), median(ts), art.cold, dv)
        end
        @printf("%-38s K=%-5d min %.3fs med %.3fs cold %.3fs  pred_var max|Δ| %.1e\n", basename(f), art.K, minimum(ts), median(ts), art.cold, dv)
        flush(stdout)
    end
end
println("done -> $OUT")
