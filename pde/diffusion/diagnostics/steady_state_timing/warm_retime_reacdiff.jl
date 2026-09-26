# Steady-state (compiled, "warm") inference re-timing for the Fisher-KPP dataset timing figure
# (`figures_onepass/diffusion/training_inference_time.py`), the counterpart of
# `pde/advection/diagnostics/steady_state_timing/warm_retime_{closed_form_v3,hmc_v2}.jl`:
# the runners' `predict_time_seconds` are cold-start timings that include JIT compilation; this
# script re-times the SAME timed block per method from the saved posterior (Ours, VI: one
# `predict_moments_grid` pass) or saved chains (HMC: `predict_bpinn_stats` over the 3000 pooled
# draws) after one untimed warm-up call per code path. Same protocol as those two scripts: closed-form families
# min-of-KREP=7 repetitions with the Ours/VI measurement order alternated per trajectory (ABAB), HMC
# min-of-3; GC flushed before each artifact; every re-timed path must reproduce the artifact's
# saved `pred_var` to machine precision (`pred_var_maxdiff` column). Run on an idle machine with
# OPENBLAS_NUM_THREADS=1. The problem and eigenbasis are rebuilt per trajectory (the Kx rule is
# IC-dependent here), exactly as the runners do.
#
# Usage: julia --project=. pde/diffusion/diagnostics/steady_state_timing/warm_retime_reacdiff.jl <out.csv> [max_cells]
# CSV columns: artifact,family,nu,rho,idx,K,warm_predict_seconds,warm_median_seconds,cold_predict_seconds,pred_var_maxdiff

include(joinpath(@__DIR__, "..", "..", "..", "vi_model.jl"))
using .VIPINN
using .VIPINN.BPINN
using .VIPINN.BPINN.GDPINN.MPPINN
using .VIPINN.BPINN.GDPINN.MPPINN.Factors
using .VIPINN.BPINN.GDPINN.MPPINN.Factors.DistributionCollections
using .VIPINN.BPINN.GDPINN.MPPINN.Factors.Gaussian
using Random, HDF5, Printf, Statistics, LinearAlgebra

BLAS.set_num_threads(1)
REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
OUT = ARGS[1]
MAX_CELLS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : typemax(Int)
KREP = 7
KREP_HMC = 3
PW_ALPHA = 0.9
NUS = [0.5, 1.0, 2.0, 5.0]
RHOS = [1.0, 2.0, 5.0, 10.0]
IDXS = [9394, 898, 2398, 5906, 2343]
PREFIX = Dict("onepass" => "mp_pinn_reacdiff_ds2", "vi" => "vi_reacdiff", "hmc" => "bpinn_mc_reacdiff")
fmt(x) = isinteger(x) ? string(Int(x)) * ".0" : string(x)
art_path(fam, nu, rho, idx) = joinpath(REPO, "artifacts", "$(PREFIX[fam])_nu$(fmt(nu))_rho$(fmt(rho))_idx$(idx).h5")

include(joinpath(REPO, "pde", "diffusion", "reacdiff_lib.jl"))

function problem_and_V(nu, rho, idx)
    prob = reacdiff_dataset_problem(nu, rho, idx; Kt = 15, time_fill = 0.5)
    g = build_reacdiff_graph(prob.p, prob.nu_norm, prob.rho_norm, MersenneTwister(7); n_ic = 512, nt = 31, nx = 48,
                             σw = 1.0, β_ic = 0.02, β_pde = 0.05, α = PW_ALPHA)
    return prob.p, g.V
end

function load_art(fam, f)
    h5open(f, "r") do h
        cold = Float64(read(attributes(h)["predict_time_seconds"]))
        pv = Float64.(read(h, "pred_var"))
        fp = Float64.(read(h, "v_fingerprint"))
        if fam == "hmc"
            nch = Int(read(attributes(h)["n_chains"]))
            samples = Vector{Vector{Float64}}()
            for c in 1:nch
                M = Float64.(read(h, "samples_chain$c"))
                append!(samples, [M[:, j] for j in 1:size(M, 2)])
            end
            return (cold = cold, pv = pv, fp = fp, samples = samples, K = length(samples[1]))
        else
            μ = Float64.(read(h, "w_eigen_mean")); σ = Float64.(read(h, "w_eigen_std"))
            return (cold = cold, pv = pv, fp = fp, μ = μ, σ = σ, K = length(μ))
        end
    end
end

function timed_block(fam, p, V, art)
    if fam == "hmc"
        samples = art.samples
        w_mean = sum(samples) ./ length(samples)
        w = V * w_mean
        samples_mat = reduce(hcat, samples)
        w_eigen_std = vec(std(samples_mat, dims = 2))
        return predict_bpinn_stats(p, V, samples; α = PW_ALPHA)
    else
        w = V * art.μ
        return predict_moments_grid(V, p, art.μ, art.σ .^ 2; α = PW_ALPHA)
    end
end

function measure(fam, p, V, art, krep)
    GC.gc()
    ts = Float64[]; local pu, pv
    for _ in 1:krep
        t = @elapsed (pu, pv) = timed_block(fam, p, V, art); push!(ts, t)
    end
    return minimum(ts), median(ts), maximum(abs.(pv .- art.pv))
end

open(OUT, "w") do io
    println(io, "artifact,family,nu,rho,idx,K,warm_predict_seconds,warm_median_seconds,cold_predict_seconds,pred_var_maxdiff")
end
warmed = Dict("onepass" => false, "vi" => false, "hmc" => false)
n_cells = 0
for nu in NUS, rho in RHOS
    global n_cells += 1
    n_cells > MAX_CELLS && break
    for (j, idx) in enumerate(IDXS)
        files = Dict(fam => art_path(fam, nu, rho, idx) for fam in keys(PREFIX))
        all(isfile, values(files)) || (println("skip nu=$nu rho=$rho idx=$idx: missing artifact"); continue)
        p, V = problem_and_V(nu, rho, idx)
        arts = Dict(fam => load_art(fam, f) for (fam, f) in files)
        for (fam, a) in arts
            maximum(abs.(a.fp .- v_fingerprint(V))) <= 1e-9 || error("eigenbasis fingerprint mismatch for $fam nu=$nu rho=$rho idx=$idx")
        end
        for fam in ("onepass", "vi", "hmc")
            warmed[fam] || (timed_block(fam, p, V, arts[fam]); warmed[fam] = true)
        end
        order = isodd(j) ? ("onepass", "vi") : ("vi", "onepass")   # ABAB alternation across trajectories
        for fam in (order..., "hmc")
            tmin, tmed, dv = measure(fam, p, V, arts[fam], fam == "hmc" ? KREP_HMC : KREP)
            open(OUT, "a") do io
                @printf(io, "%s,%s,%g,%g,%d,%d,%.4f,%.4f,%.4f,%.3e\n", basename(files[fam]), fam, nu, rho, idx, arts[fam].K,
                        tmin, tmed, arts[fam].cold, dv)
            end
            @printf("%-48s K=%-4d min %.3fs med %.3fs cold %.3fs  pred_var max|Δ| %.1e\n", basename(files[fam]), arts[fam].K,
                    tmin, tmed, arts[fam].cold, dv)
            flush(stdout)
        end
    end
end
println("done -> $OUT")
