# ============================================================================
# s40: Multi-Start Benchmark — Statistical Metric Comparison
# ============================================================================
#
# Goal:   Compare metrics over N random initial points. Produces Table 1.
# Output: results/benchmark/raw.csv, results/benchmark/summary.csv
#         results/logs/s40_random_benchmark_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s40_random_benchmark.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s40_random_benchmark")

# ── Configuration ───────────────────────────────────────────────────────

const κ = 50.0
const ϕ = π / 6
const δ = 0.1
const α_base = 0.8
const λ = 1.0
const T_final = 50.0
const tol = 1e-6
const xstar = [0.5, 0.3]

const N_RANDOM = 50
const SEED = 1234
const METRICS = [:identity, :Qinv, :diag_inv]
const METRIC_LABELS = Dict(:identity => "euclid", :Qinv => "M=Q^-1", :diag_inv => "M=diag_inv")

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "benchmark")
mkpath(results_dir)

# ── Generate initial points ─────────────────────────────────────────────

rng = MersenneTwister(SEED)
random_x0s = [2.0 * rand(rng, 2) .- 1.0 for _ in 1:N_RANDOM]

# ── Main benchmark ──────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Multi-Start Benchmark: N=$(N_RANDOM) random starts")
@printf(tee, "  κ=%.0f, δ=%.2f, α_base=%.2f, T=%.0f\n", κ, δ, α_base, T_final)
println(tee, "  Metrics: ", join(values(METRIC_LABELS), ", "))
println(tee, "=" ^ 70)

all_results = Dict{Symbol, Vector{NamedTuple}}()

raw_path = joinpath(results_dir, "raw.csv")
raw_io = open(raw_path, "w")
println(raw_io, "metric,start_idx,x0_1,x0_2,alpha,r_final,V_final,t_tol2,t_tol6,converged,kappa_M")

for met in METRICS
    prob_tmp = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
    α_met = α_base / opnorm(prob_tmp.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)
    println(tee, "\n─── Metric: $(METRIC_LABELS[met])  α=$(round(α_met, sigdigits=4)) ───")
    met_results = NamedTuple[]

    for (idx, x0) in enumerate(random_x0s)
        prob = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
        prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                              M=prob.M, x0=x0, n=prob.n, name=prob.name)

        ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=1.0, xstar=xstar)

        t2 = time_to_tol(ts, rs, 1e-2)
        t6 = time_to_tol(ts, rs, 1e-6)
        r_final = rs[end]
        V_final = Vs[end]
        conv = r_final < tol
        κ_M = cond(prob.M)

        @printf(raw_io, "%s,%d,%.6f,%.6f,%.4f,%.6e,%.6e,%.4f,%.4f,%s,%.2f\n",
                met, idx, x0[1], x0[2], α_met, r_final, V_final, t2, t6, conv, κ_M)
        flush(raw_io)

        push!(met_results, (r_final=r_final, V_final=V_final, t2=t2, t6=t6, conv=conv))
    end

    all_results[met] = met_results
end
close(raw_io)

# ── Summary statistics ──────────────────────────────────────────────────

println(tee, "\n" * "=" ^ 70)
println(tee, "Summary Statistics")
println(tee, "=" ^ 70)

@printf(tee, "%-10s %8s %12s %12s %8s %10s\n",
        "Metric", "Success", "Med r_final", "Med V_final", "κ_M", "Med t(1e-2)")
println(tee, "-" ^ 62)

summary_path = joinpath(results_dir, "summary.csv")
summary_io = open(summary_path, "w")
println(summary_io, "metric,n_starts,n_success,success_rate,median_r,median_V,kappa_M,median_t2")

for met in METRICS
    prob_tmp2 = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
    κ_M = cond(prob_tmp2.M)
    res = all_results[met]

    n_success = count(r -> r.r_final < 1e-2, res)
    sr = n_success / N_RANDOM * 100
    med_r = median([r.r_final for r in res])
    med_V = median([r.V_final for r in res])
    finite_t2 = filter(isfinite, [r.t2 for r in res])
    med_t2 = isempty(finite_t2) ? Inf : median(finite_t2)

    @printf(tee, "%-10s %7.1f%% %12.2e %12.2e %8.1f %10.1f\n",
            METRIC_LABELS[met], sr, med_r, med_V, κ_M, med_t2)
    @printf(summary_io, "%s,%d,%d,%.4f,%.6e,%.6e,%.2f,%.4f\n",
            met, N_RANDOM, n_success, sr/100, med_r, med_V, κ_M, med_t2)
end
close(summary_io)

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $results_dir")
println(tee, "  raw.csv — per-start data")
println(tee, "  summary.csv — aggregated statistics")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
