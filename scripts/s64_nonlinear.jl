# ============================================================================
# s64: Nonlinear Monotone Operator — Metric Comparison
# ============================================================================
#
# Goal:   Test Problem 7 (nonlinear monotone QVI, n=5) with identity and Q⁻¹
#         metrics. Compare convergence speed and trajectory structure.
# Output: results/nonlinear_monotone/ — trajectory CSVs + summary
#         results/logs/s64_nonlinear_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s64_nonlinear.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s64_nonlinear")

# ── Configuration ───────────────────────────────────────────────────────

const n_dim = 5
const δ = 0.1
const α_base = 0.5
const λ = 1.0
const T_final = 80.0
const save_dt = 0.2
const tol = 1e-6
const xstar = [1.0, 2.0, 3.0, 2.0, 1.0]   # known solution

const METRICS = [:identity, :Qinv]
const METRIC_LABELS = Dict(
    :identity => "Euclidean",
    :Qinv     => "M=Q^{-1}",
)

# Multiple initial points
const INIT_POINTS = [
    [4.5, 0.5, 4.5, 0.5, 4.5],    # alternating near boundaries
    [0.1, 0.1, 0.1, 0.1, 0.1],    # near lower bound
    [4.9, 4.9, 4.9, 4.9, 4.9],    # near upper bound
    [2.5, 2.5, 2.5, 2.5, 2.5],    # center of box
    [1.0, 4.0, 1.0, 4.0, 1.0],    # another alternating pattern
    [0.5, 1.5, 2.5, 3.5, 4.5],    # diagonal across box
]
const INIT_LABELS = ["alt_boundary", "lower", "upper", "center", "alt_2", "diagonal"]

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "nonlinear_monotone")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Problem 7: Nonlinear Monotone Operator (n=$(n_dim))")
@printf(tee, "  F(x) = Qx + rho*atan(x-2) + q, rho=1.0\n")
@printf(tee, "  x_bar = %s\n", string(xstar))
@printf(tee, "  S = [0,5]^%d, delta=%.2f, alpha_base=%.2f, T=%.0f\n",
        n_dim, δ, α_base, T_final)
println(tee, "  Metrics: ", join([METRIC_LABELS[m] for m in METRICS], ", "))
println(tee, "  Initial points: $(length(INIT_POINTS))")
println(tee, "=" ^ 70)

summary_lines = String[]
push!(summary_lines, "metric,x0_label,alpha,norm_M,cond_M,t_tol_2,t_tol_6,r_final,V_final,err_final,iters,status")

for met in METRICS
    prob_tmp = get_problem(7; n=n_dim, δ=δ, metric=met)
    α_met = α_base / opnorm(prob_tmp.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)
    norm_M = opnorm(prob_tmp.M)
    cond_M = cond(prob_tmp.M)
    println(tee, @sprintf("\n--- Metric: %-15s  alpha=%.6f  ||M||=%.4e  cond(M)=%.2f ---",
            METRIC_LABELS[met], α_met, norm_M, cond_M))

    for (idx, x0) in enumerate(INIT_POINTS)
        prob = get_problem(7; n=n_dim, δ=δ, metric=met)
        prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                              M=prob.M, x0=x0, n=prob.n, name=prob.name)

        ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=save_dt, xstar=xstar)

        # Save trajectory
        traj_file = joinpath(results_dir, "traj_$(met)_x0_$(idx).csv")
        open(traj_file, "w") do io
            println(io, "t," * join(["x$i" for i in 1:n_dim], ",") * ",residual,V,error_norm")
            for k in eachindex(ts)
                xvals = join([@sprintf("%.10e", xs[k][i]) for i in 1:n_dim], ",")
                err = norm(xs[k] - xstar)
                @printf(io, "%.6f,%s,%.10e,%.10e,%.10e\n",
                        ts[k], xvals, rs[k], Vs[k], err)
            end
        end

        t2 = time_to_tol(ts, rs, 1e-2)
        t6 = time_to_tol(ts, rs, 1e-6)
        r_final = rs[end]
        V_final = Vs[end]
        err_final = norm(xs[end] - xstar)
        status = r_final < tol ? "converged" : "not_converged"

        push!(summary_lines, @sprintf("%s,%s,%.6f,%.4e,%.2f,%.4f,%.4f,%.4e,%.4e,%.4e,%d,%s",
              met, INIT_LABELS[idx], α_met, norm_M, cond_M,
              t2, t6, r_final, V_final, err_final, length(ts), status))

        @printf(tee, "  x0 %-15s  r=%.2e  ||x-x*||=%.2e  t(1e-2)=%6.1f  t(1e-6)=%6.1f  %s\n",
                INIT_LABELS[idx], r_final, err_final, t2, t6, status)
    end
end

# ── Metric comparison summary ─────────────────────────────────────────

println(tee, "\n--- Metric Comparison Summary ---")
@printf(tee, "%-15s %10s %10s %12s\n", "Metric", "Med t(1e-2)", "Med t(1e-6)", "Med ||x-x*||")
println(tee, "-" ^ 50)

for met in METRICS
    prob_tmp = get_problem(7; n=n_dim, δ=δ, metric=met)
    α_met = α_base / opnorm(prob_tmp.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)

    t2_list = Float64[]
    t6_list = Float64[]
    err_list = Float64[]

    for x0 in INIT_POINTS
        prob = get_problem(7; n=n_dim, δ=δ, metric=met)
        prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                              M=prob.M, x0=x0, n=prob.n, name=prob.name)
        ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=1.0, xstar=xstar)
        push!(t2_list, time_to_tol(ts, rs, 1e-2))
        push!(t6_list, time_to_tol(ts, rs, 1e-6))
        push!(err_list, norm(xs[end] - xstar))
    end

    med_t2 = median(filter(isfinite, t2_list))
    med_t6 = median(filter(isfinite, t6_list))
    med_err = median(err_list)
    @printf(tee, "%-15s %10.2f %10.2f %12.4e\n", METRIC_LABELS[met], med_t2, med_t6, med_err)
end

# ── Save summary ────────────────────────────────────────────────────────

summary_file = joinpath(results_dir, "summary.csv")
open(summary_file, "w") do io
    for line in summary_lines
        println(io, line)
    end
end

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $results_dir")
println(tee, "  Trajectory files: traj_{metric}_x0_{idx}.csv")
println(tee, "  Summary: summary.csv")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
