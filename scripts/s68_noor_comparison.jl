# ============================================================================
# s68: Noor (2003) Example 4.1 — QVI on ℝ²₊
# ============================================================================
#
# Goal:   Test Problem 9 (Noor 2003, Example 4.1) from multiple initial points,
#         including some starting outside ℝ²₊ to test projection behavior.
#         Known solution: x̄ = (0, 0).
# Output: results/noor_comparison/ — trajectory CSVs + summary
#         results/logs/s68_noor_comparison_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s68_noor_comparison.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s68_noor_comparison")

# ── Configuration ───────────────────────────────────────────────────────

const α_base = 0.5
const λ = 1.0
const T_final = 30.0
const save_dt = 0.05
const tol = 1e-8

# Known solution
const xstar = [0.0, 0.0]

# Multiple initial points — some outside ℝ²₊ to test projection
const INIT_POINTS = [
    [1.0, 1.0],
    [2.0, 3.0],
    [5.0, 5.0],
    [0.5, 2.0],
    [3.0, 0.1],
    [-0.5, 1.0],   # x₁ < 0: starts outside ℝ²₊
]
const INIT_LABELS = ["(1,1)", "(2,3)", "(5,5)", "(0.5,2)", "(3,0.1)", "(-0.5,1)"]

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "noor_comparison")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Problem 9: Noor (2003) Example 4.1 — QVI on R²₊")
println(tee, "  F(u) = (u₁+u₂+sin(u₁), u₁+u₂+sin(u₂))")
println(tee, "  S = R²₊, m(u) = u/8, K_m = 0.125")
println(tee, "  Known solution: x* = (0, 0)")
@printf(tee, "  alpha_base=%.2f, T=%.0f, tol=%.0e\n", α_base, T_final, tol)
println(tee, "  Initial points: $(length(INIT_POINTS))")
println(tee, "=" ^ 70)

prob_ref = get_problem(9)
α_met = α_base / opnorm(prob_ref.M)
cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)

println(tee, @sprintf("\n  alpha=%.6f  ||M||=%.4e", α_met, opnorm(prob_ref.M)))
println(tee, "")

summary_lines = String[]
push!(summary_lines, "x0_idx,x0_label,x0_1,x0_2,alpha,r_final,x_final_1,x_final_2,dist_to_xstar,t_tol_2,t_tol_6,t_tol_8,iters,status")

final_points = Vector{Float64}[]

for (idx, x0) in enumerate(INIT_POINTS)
    prob = get_problem(9)
    prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                          M=prob.M, x0=x0, n=prob.n, name=prob.name)

    ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=save_dt)

    # Save trajectory
    traj_file = joinpath(results_dir, "traj_x0_$(idx).csv")
    open(traj_file, "w") do io
        println(io, "t,x1,x2,residual,dist_to_xstar")
        for k in eachindex(ts)
            dist = norm(xs[k] - xstar)
            @printf(io, "%.6f,%.10e,%.10e,%.10e,%.10e\n",
                    ts[k], xs[k][1], xs[k][2], rs[k], dist)
        end
    end

    t2 = time_to_tol(ts, rs, 1e-2)
    t6 = time_to_tol(ts, rs, 1e-6)
    t8 = time_to_tol(ts, rs, 1e-8)
    r_final = rs[end]
    x_final = xs[end]
    dist_final = norm(x_final - xstar)
    status = r_final < tol ? "converged" : "not_converged"
    push!(final_points, x_final)

    push!(summary_lines, @sprintf("%d,%s,%.4f,%.4f,%.6f,%.4e,%.10f,%.10f,%.4e,%.4f,%.4f,%.4f,%d,%s",
          idx, INIT_LABELS[idx], x0[1], x0[2], α_met, r_final,
          x_final[1], x_final[2], dist_final, t2, t6, t8, length(ts), status))

    @printf(tee, "  x0 #%d %-10s  ->  x*=[%9.6f,%9.6f]  ||x-x*||=%.2e  r=%.2e  t(1e-2)=%5.1f  t(1e-6)=%5.1f  t(1e-8)=%5.1f  %s\n",
            idx, INIT_LABELS[idx], x_final[1], x_final[2], dist_final, r_final, t2, t6, t8, status)
end

# ── Convergence verification ──────────────────────────────────────────

println(tee, "\n--- Convergence Verification ---")
all_converged = all(norm(fp - xstar) < 1e-4 for fp in final_points)
@printf(tee, "  All converged to x* = (0,0): %s\n", all_converged ? "YES" : "NO")
for (idx, fp) in enumerate(final_points)
    @printf(tee, "    x0 #%d: ||x_final - x*|| = %.4e\n", idx, norm(fp - xstar))
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
println(tee, "  Trajectory files: traj_x0_{idx}.csv")
println(tee, "  Summary: summary.csv")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
