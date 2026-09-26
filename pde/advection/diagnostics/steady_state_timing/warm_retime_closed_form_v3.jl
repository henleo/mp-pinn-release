# Steady-state re-timing for the CLOSED-FORM families only (onepass, vi): same reconstruction/checks
# as `warm_retime_hmc_v2.jl`, but (i) no HMC bursts interleaved between sub-second measurements,
# (ii) min-of-KREP=7 repetitions, (iii) measurement
# order of Ours vs VI alternated per trajectory (ABAB) so neither family systematically follows the other.
# HMC rows come from `warm_retime_hmc_v2.jl` (min-of-3) -- its 15-30 s timings are not noise-limited.
# Usage: julia --project=. pde/advection/diagnostics/steady_state_timing/warm_retime_closed_form_v3.jl <out.csv> [beta]
include(joinpath(@__DIR__, "..", "..", "..", "vi_model.jl"))
using .VIPINN
using .VIPINN.BPINN.GDPINN.MPPINN
using Random, HDF5, Printf, Statistics, LinearAlgebra

REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
OUT = ARGS[1]
KREP = 7
PW_ALPHA = 0.9
# Optional second argument: a single wave speed to re-time (e.g. 7.0); default = all five.
BETAS = length(ARGS) >= 2 ? [parse(Float64, ARGS[2])] : [0.1, 0.4, 1.0, 2.0, 7.0]
IDXS = [9394, 898, 2398, 5906, 2343, 8225, 5506, 6451, 2670, 3497]
PREFIX = Dict("onepass" => "mp_pinn_beta", "vi" => "vi_beta", "hmc" => "bpinn_mc_beta")
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
    w = V * art.μ
    return predict_moments_grid(V, p, art.μ, art.σ .^ 2; α = PW_ALPHA)
end

function load_art(fam, f)
    h5open(f, "r") do h
        Kt = Int(read(attributes(h)["Kt"])); cold = Float64(read(attributes(h)["predict_time_seconds"]))
        pv = Float64.(read(h, "pred_var")); μ = Float64.(read(h, "w_eigen_mean")); σ = Float64.(read(h, "w_eigen_std"))
        return (Kt = Kt, cold = cold, pv = pv, μ = μ, σ = σ, K = length(μ))
    end
end

function measure!(io, fam, beta, idx, p, V)
    f = art_path(fam, beta, idx); isfile(f) || return
    art = load_art(fam, f)
    GC.gc()
    ts = Float64[]; local pu, pv
    for r in 1:KREP
        t = @elapsed (pu, pv) = timed_block(fam, p, V, art); push!(ts, t)
    end
    dv = maximum(abs.(pv .- art.pv))
    @printf(io, "%s,%s,%g,%d,%d,%.4f,%.4f,%.4f,%.3e\n", basename(f), fam, beta, idx, art.K, minimum(ts), median(ts), art.cold, dv)
    @printf("%-38s K=%-5d min %.3fs med %.3fs cold %.3fs  pred_var max|Δ| %.1e\n", basename(f), art.K, minimum(ts), median(ts), art.cold, dv)
    flush(stdout)
end

open(OUT, "w") do io
    println(io, "artifact,family,beta,idx,K,warm_predict_seconds,warm_median_seconds,cold_predict_seconds,pred_var_maxdiff")
end
warmed = false
for beta in BETAS
    common = [i for i in IDXS if all(isfile(art_path(f, beta, i)) for f in ("onepass", "vi", "hmc"))]
    println("beta=$beta common idxs: $common")
    for (n, idx) in enumerate(common)
        art0 = load_art("onepass", art_path("onepass", beta, idx))
        p, V = problem_and_V(beta, idx, art0.Kt)
        if !warmed; timed_block("onepass", p, V, art0); global warmed = true; end
        order = isodd(n) ? ("onepass", "vi") : ("vi", "onepass")   # alternate ABAB
        open(OUT, "a") do io
            for fam in order; measure!(io, fam, beta, idx, p, V); end
        end
    end
end
println("done -> $OUT")
