# ============================================================================
# s66: Random High-Dimensional QVI — Scaling Study
# ============================================================================
#
# Goal:   Test Problem 8 (random high-dim QVI) for n=10,20,50 with three
#         metrics (identity, Dinv, diag_inv). Report convergence statistics.
# Output: results/scaling/ — per-dimension CSVs + summary
#         results/logs/s66_scaling_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s66_scaling.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s66_scaling")

# ── Configuration ───────────────────────────────────────────────────────

const DIMS = [10, 20, 50]
const δ = 0.1
const α_base = 0.3
const λ = 1.0
const T_final = 100.0
const save_dt = 1.0
const tol = 1e-6
const N_RANDOM = 10       # random initial points per (n, metric)
const SEED_X0 = 9999

const METRICS = [:identity, :Dinv, :diag_inv]
const METRIC_LABELS = Dict(
    :identity => "Euclidean",
    :Dinv     => "M=D^{-1}",
    :diag_inv => "M=diag(Q)^{-1}",
)

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "scaling")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Problem 8: Random High-Dimensional QVI — Scaling Study")
@printf(tee, "  Dimensions: %s\n", string(DIMS))
@printf(tee, "  delta=%.2f, alpha_base=%.2f, T=%.0f, tol=%.0e\n", δ, α_base, T_final, tol)
@printf(tee, "  Random starts per (n, metric): %d\n", N_RANDOM)
println(tee, "  Metrics: ", join([METRIC_LABELS[m] for m in METRICS], ", "))
println(tee, "=" ^ 70)

# Raw results file
raw_path = joinpath(results_dir, "raw.csv")
raw_io = open(raw_path, "w")
println(raw_io, "n,metric,start_idx,alpha,norm_M,cond_M,r_final,t_tol_2,t_tol_6,converged,wall_s")

# Summary accumulator: (n, metric) -> vector of named tuples
all_results = Dict{Tuple{Int,Symbol}, Vector{NamedTuple}}()

for n_dim in DIMS
    println(tee, "\n" * "=" ^ 70)
    println(tee, "  Dimension n = $(n_dim)")
    println(tee, "=" ^ 70)

    # Generate random initial points for this dimension
    rng_x0 = MersenneTwister(SEED_X0)
    random_x0s = [10.0 * rand(rng_x0, n_dim) for _ in 1:N_RANDOM]

    for met in METRICS
        prob_tmp = get_problem(8; n=n_dim, δ=δ, metric=met)
        α_met = α_base / opnorm(prob_tmp.M)
        cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)
        norm_M = opnorm(prob_tmp.M)
        cond_M = cond(prob_tmp.M)

        println(tee, @sprintf("\n  --- n=%d, Metric: %-20s  alpha=%.6f  ||M||=%.4e  cond(M)=%.2f ---",
                n_dim, METRIC_LABELS[met], α_met, norm_M, cond_M))

        met_results = NamedTuple[]

        for (idx, x0) in enumerate(random_x0s)
            prob = get_problem(8; n=n_dim, δ=δ, metric=met)
            prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                                  M=prob.M, x0=x0, n=prob.n, name=prob.name)

            wall_start = time()
            ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=save_dt)
            wall_s = time() - wall_start

            t2 = time_to_tol(ts, rs, 1e-2)
            t6 = time_to_tol(ts, rs, 1e-6)
            r_final = rs[end]
            conv = r_final < tol

            push!(met_results, (r_final=r_final, t2=t2, t6=t6, conv=conv, wall_s=wall_s))

            @printf(raw_io, "%d,%s,%d,%.6f,%.4e,%.2f,%.6e,%.4f,%.4f,%s,%.3f\n",
                    n_dim, met, idx, α_met, norm_M, cond_M, r_final, t2, t6, conv, wall_s)
            flush(raw_io)
        end

        all_results[(n_dim, met)] = met_results

        # Per-metric stats
        n_conv = count(r -> r.conv, met_results)
        med_r = median([r.r_final for r in met_results])
        finite_t6 = filter(isfinite, [r.t6 for r in met_results])
        med_t6 = isempty(finite_t6) ? Inf : median(finite_t6)
        finite_t2 = filter(isfinite, [r.t2 for r in met_results])
        med_t2 = isempty(finite_t2) ? Inf : median(finite_t2)
        med_wall = median([r.wall_s for r in met_results])

        @printf(tee, "    Conv: %d/%d  Med r=%.2e  Med t(1e-2)=%.1f  Med t(1e-6)=%.1f  Med wall=%.2fs\n",
                n_conv, N_RANDOM, med_r, med_t2, med_t6, med_wall)
    end
end
close(raw_io)

# ── Summary table ──────────────────────────────────────────────────────

println(tee, "\n" * "=" ^ 70)
println(tee, "SUMMARY TABLE: Convergence Statistics")
println(tee, "=" ^ 70)

@printf(tee, "%-5s %-15s %8s %12s %12s %12s %10s\n",
        "n", "Metric", "Conv", "Med r_final", "Med t(1e-2)", "Med t(1e-6)", "Med wall")
println(tee, "-" ^ 78)

summary_path = joinpath(results_dir, "summary.csv")
summary_io = open(summary_path, "w")
println(summary_io, "n,metric,n_starts,n_conv,conv_rate,median_r,median_t2,median_t6,median_wall_s")

for n_dim in DIMS
    for met in METRICS
        key = (n_dim, met)
        haskey(all_results, key) || continue
        res = all_results[key]

        n_conv = count(r -> r.conv, res)
        sr = n_conv / N_RANDOM * 100
        med_r = median([r.r_final for r in res])
        finite_t2 = filter(isfinite, [r.t2 for r in res])
        med_t2 = isempty(finite_t2) ? Inf : median(finite_t2)
        finite_t6 = filter(isfinite, [r.t6 for r in res])
        med_t6 = isempty(finite_t6) ? Inf : median(finite_t6)
        med_wall = median([r.wall_s for r in res])

        @printf(tee, "%-5d %-15s %7.0f%% %12.2e %12.1f %12.1f %10.2f\n",
                n_dim, METRIC_LABELS[met], sr, med_r, med_t2, med_t6, med_wall)
        @printf(summary_io, "%d,%s,%d,%d,%.4f,%.6e,%.4f,%.4f,%.3f\n",
                n_dim, met, N_RANDOM, n_conv, sr/100, med_r, med_t2, med_t6, med_wall)
    end
    println(tee, "-" ^ 78)
end
close(summary_io)

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $results_dir")
println(tee, "  raw.csv — per-start data")
println(tee, "  summary.csv — aggregated statistics")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
