# ============================================================================
# s20: Example 1 — 2D Affine QVI: Trajectory Geometry Under Different Metrics
# ============================================================================
#
# Goal:   Visualize how metric choice reshapes QVI trajectories.
#         This is the flagship visual example for the paper.
# Output: results/example1/ — CSV files + summary
#         results/logs/s20_example1_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s20_example1.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s20_example1")

# ── Configuration ───────────────────────────────────────────────────────

const κ = 50.0
const ϕ = π / 6
const δ = 0.1
const α_base = 0.8      # base step-size: α = α_base / (‖M‖ · K_F)
const λ = 1.0
const T_final = 50.0
const save_dt = 0.05
const tol = 1e-6
const xstar = [0.5, 0.3]

const METRICS = [:identity, :Qinv, :diag_inv]
const METRIC_LABELS = Dict(
    :identity => "Euclidean",
    :Qinv     => "M=Q^{-1}",
    :diag_inv => "M=diag(1/κ,1)",
)

const INIT_POINTS = [
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]
]

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "example1")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Example 1: 2D Affine QVI — Trajectory Geometry")
@printf(tee, "  κ=%.0f, ϕ=π/%.0f, δ=%.2f, α_base=%.2f, T=%.0f\n", κ, π/ϕ, δ, α_base, T_final)
@printf(tee, "  x̄ = %s\n", string(xstar))
println(tee, "  Metrics: ", join([METRIC_LABELS[m] for m in METRICS], ", "))
println(tee, "  Initial points: $(length(INIT_POINTS))")
println(tee, "=" ^ 70)

summary_lines = String[]
push!(summary_lines, "metric,x0_idx,x0,alpha,t_tol_6,t_tol_2,r_final,V_final,iters,status")

for met in METRICS
    prob_tmp = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
    # Step-size: α = α_base / ‖M‖  (keeps α·‖M‖ constant across metrics)
    α_met = α_base / opnorm(prob_tmp.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)
    println(tee, "\n─── Metric: $(METRIC_LABELS[met])  α=$(round(α_met, sigdigits=4)) ───")

    for (idx, x0) in enumerate(INIT_POINTS)
        prob = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
        prob_with_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                                   M=prob.M, x0=x0, n=prob.n, name=prob.name)

        ts, xs, rs, Vs = solve_qvi_diffeq(prob_with_x0, cfg; save_dt=save_dt, xstar=xstar)

        traj_file = joinpath(results_dir, "traj_$(met)_x0_$(idx).csv")
        open(traj_file, "w") do io
            println(io, "t,x1,x2,residual,V")
            for k in eachindex(ts)
                @printf(io, "%.6f,%.10e,%.10e,%.10e,%.10e\n",
                        ts[k], xs[k][1], xs[k][2], rs[k], Vs[k])
            end
        end

        t6 = time_to_tol(ts, rs, 1e-6)
        t2 = time_to_tol(ts, rs, 1e-2)
        r_final = rs[end]
        V_final = Vs[end]
        status = r_final < tol ? "converged" : "not_converged"

        push!(summary_lines, @sprintf("%s,%d,[%.1f;%.1f],%.4f,%.4f,%.4f,%.4e,%.4e,%d,%s",
              met, idx, x0[1], x0[2], α_met, t6, t2, r_final, V_final, length(ts), status))

        @printf(tee, "  x0 #%d [%5.1f,%5.1f]  r_final=%.2e  V_final=%.2e  t(1e-2)=%.1f  t(1e-6)=%.1f\n",
                idx, x0[1], x0[2], r_final, V_final, t2, t6)
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
println(tee, "  Summary: summary.csv")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
