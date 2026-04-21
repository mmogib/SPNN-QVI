# ============================================================================
# s60: Discretized Obstacle QVI — 1D Membrane Problem
# ============================================================================
#
# Goal:   Test Problem 5 (obstacle QVI) with n=20 interior grid points,
#         three metrics (identity, Ainv, Jacobi), and multiple initial points.
# Output: results/obstacle/ — trajectory CSVs + summary
#         results/logs/s60_obstacle_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s60_obstacle.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s60_obstacle")

# ── Configuration ───────────────────────────────────────────────────────

const n_grid = 20
const δ = 0.1
const α_base = 0.8
const λ = 1.0
const T_final = 100.0
const save_dt = 0.5
const tol = 1e-6

const METRICS = [:identity, :Ainv, :jacobi]
const METRIC_LABELS = Dict(
    :identity => "Euclidean",
    :Ainv     => "M=A^{-1}",
    :jacobi   => "M=diag(A)^{-1}",
)

# ── Initial points ─────────────────────────────────────────────────────

h = 1.0 / (n_grid + 1)
grid = [i * h for i in 1:n_grid]
ψ0 = 0.2 * sin.(π * grid)

const INIT_POINTS = [
    0.5 * (ψ0 .+ 1.0),                          # midway between obstacle and 1
    ψ0 .+ 0.01,                                   # just above obstacle
    ones(n_grid),                                  # flat at 1
    0.5 * ones(n_grid),                            # flat at 0.5
]
const INIT_LABELS = ["midway", "near_obstacle", "flat_1", "flat_0.5"]

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "obstacle")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Problem 5: Discretized Obstacle QVI — 1D Membrane")
@printf(tee, "  n=%d, h=%.4f, delta=%.2f, alpha_base=%.2f, T=%.0f\n",
        n_grid, h, δ, α_base, T_final)
println(tee, "  Metrics: ", join([METRIC_LABELS[m] for m in METRICS], ", "))
println(tee, "  Initial points: $(length(INIT_POINTS))")
println(tee, "=" ^ 70)

summary_lines = String[]
push!(summary_lines, "metric,x0_label,alpha,norm_M,t_tol_2,t_tol_6,r_final,iters,status")

for met in METRICS
    prob_tmp = get_problem(5; n=n_grid, δ=δ, metric=met)
    α_met = α_base / opnorm(prob_tmp.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)
    norm_M = opnorm(prob_tmp.M)
    println(tee, @sprintf("\n--- Metric: %-20s  alpha=%.6f  ||M||=%.4e ---",
            METRIC_LABELS[met], α_met, norm_M))

    for (idx, x0) in enumerate(INIT_POINTS)
        prob = get_problem(5; n=n_grid, δ=δ, metric=met)
        prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                              M=prob.M, x0=x0, n=prob.n, name=prob.name)

        ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=save_dt)

        # Save trajectory (save selected components + residual)
        traj_file = joinpath(results_dir, "traj_$(met)_x0_$(idx).csv")
        open(traj_file, "w") do io
            # Header: t, x1, x_n/4, x_n/2, x_3n/4, x_n, residual
            comps = [1, max(1, n_grid÷4), n_grid÷2, max(1, 3*n_grid÷4), n_grid]
            hdr = "t," * join(["x$(c)" for c in comps], ",") * ",residual"
            println(io, hdr)
            for k in eachindex(ts)
                vals = join([@sprintf("%.10e", xs[k][c]) for c in comps], ",")
                @printf(io, "%.6f,%s,%.10e\n", ts[k], vals, rs[k])
            end
        end

        t2 = time_to_tol(ts, rs, 1e-2)
        t6 = time_to_tol(ts, rs, 1e-6)
        r_final = rs[end]
        status = r_final < tol ? "converged" : "not_converged"

        push!(summary_lines, @sprintf("%s,%s,%.6f,%.4e,%.4f,%.4f,%.4e,%d,%s",
              met, INIT_LABELS[idx], α_met, norm_M, t2, t6, r_final, length(ts), status))

        @printf(tee, "  x0 %-15s  r_final=%.2e  t(1e-2)=%7.2f  t(1e-6)=%7.2f  %s\n",
                INIT_LABELS[idx], r_final, t2, t6, status)
    end
end

# ── Save final profiles ───────────────────────────────────────────────

# For each metric, save the final membrane profile from the first initial point
for met in METRICS
    prob = get_problem(5; n=n_grid, δ=δ, metric=met)
    α_met = α_base / opnorm(prob.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)

    ts, xs, rs, Vs = solve_qvi_diffeq(prob, cfg; save_dt=save_dt)

    profile_file = joinpath(results_dir, "profile_$(met).csv")
    open(profile_file, "w") do io
        println(io, "grid_x,x_final,obstacle")
        for i in 1:n_grid
            @printf(io, "%.6f,%.10e,%.10e\n", grid[i], xs[end][i], ψ0[i])
        end
    end
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
println(tee, "  Membrane profiles: profile_{metric}.csv")
println(tee, "  Summary: summary.csv")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
