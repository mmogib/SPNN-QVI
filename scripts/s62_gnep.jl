# ============================================================================
# s62: GNEP — Cournot Duopoly
# ============================================================================
#
# Goal:   Test Problem 6 (Cournot duopoly GNEP) from multiple initial points.
#         Show convergence to Nash equilibrium under the QVI formulation.
# Output: results/gnep/ — trajectory CSVs + summary
#         results/logs/s62_gnep_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s62_gnep.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s62_gnep")

# ── Configuration ───────────────────────────────────────────────────────

const δ = 0.1
const α_base = 0.3
const λ = 1.0
const T_final = 50.0
const save_dt = 0.1
const tol = 1e-6

# Multiple initial points: corners and interior of the feasible region
const INIT_POINTS = [
    [1.0, 1.0],
    [8.0, 1.0],
    [1.0, 8.0],
    [6.0, 6.0],    # near coupled constraint
    [5.0, 5.0],
    [0.1, 0.1],
    [9.0, 2.0],
    [2.0, 9.0],
    [3.0, 3.0],
    [7.0, 4.0],
]

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "gnep")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Problem 6: GNEP — Cournot Duopoly")
@printf(tee, "  d=20, lambda_p=4, rho=1, c1=1, c2=2\n")
@printf(tee, "  capacity=10, total_cap=12, delta=%.2f\n", δ)
@printf(tee, "  alpha_base=%.2f, T=%.0f, tol=%.0e\n", α_base, T_final, tol)
println(tee, "  Initial points: $(length(INIT_POINTS))")
println(tee, "=" ^ 70)

summary_lines = String[]
push!(summary_lines, "x0_idx,x0_1,x0_2,alpha,r_final,x_final_1,x_final_2,t_tol_2,t_tol_6,iters,status")

prob_ref = get_problem(6; δ=δ)
α_met = α_base / opnorm(prob_ref.M)
cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)

println(tee, @sprintf("\n  alpha=%.6f  ||M||=%.4e", α_met, opnorm(prob_ref.M)))
println(tee, "")

# Track all final points to check convergence to same equilibrium
final_points = Vector{Float64}[]

for (idx, x0) in enumerate(INIT_POINTS)
    prob = get_problem(6; δ=δ)
    prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                          M=prob.M, x0=x0, n=prob.n, name=prob.name)

    ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=save_dt)

    # Save trajectory
    traj_file = joinpath(results_dir, "traj_x0_$(idx).csv")
    open(traj_file, "w") do io
        println(io, "t,x1,x2,residual")
        for k in eachindex(ts)
            @printf(io, "%.6f,%.10e,%.10e,%.10e\n",
                    ts[k], xs[k][1], xs[k][2], rs[k])
        end
    end

    t2 = time_to_tol(ts, rs, 1e-2)
    t6 = time_to_tol(ts, rs, 1e-6)
    r_final = rs[end]
    x_final = xs[end]
    status = r_final < tol ? "converged" : "not_converged"
    push!(final_points, x_final)

    push!(summary_lines, @sprintf("%d,%.4f,%.4f,%.6f,%.4e,%.10f,%.10f,%.4f,%.4f,%d,%s",
          idx, x0[1], x0[2], α_met, r_final, x_final[1], x_final[2],
          t2, t6, length(ts), status))

    @printf(tee, "  x0 #%2d [%4.1f,%4.1f]  ->  x*=[%7.4f,%7.4f]  r=%.2e  t(1e-2)=%5.1f  t(1e-6)=%5.1f  %s\n",
            idx, x0[1], x0[2], x_final[1], x_final[2], r_final, t2, t6, status)
end

# ── Check equilibrium consistency ─────────────────────────────────────

println(tee, "\n--- Equilibrium Consistency ---")
converged_pts = [fp for (fp, x0) in zip(final_points, INIT_POINTS)
                 if norm(fp) < 1e10]  # exclude diverged
if length(converged_pts) >= 2
    spread = maximum(norm(p - converged_pts[1]) for p in converged_pts)
    @printf(tee, "  Max spread among final points: %.4e\n", spread)
    @printf(tee, "  Mean equilibrium: [%.6f, %.6f]\n",
            mean(p[1] for p in converged_pts), mean(p[2] for p in converged_pts))
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
