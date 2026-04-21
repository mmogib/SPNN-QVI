# ============================================================================
# s30: Example 2 — 2D Nonlinear QVI: Interior Solution
# ============================================================================
#
# Goal:   Show metric effects with nonlinear operator and interior solution.
#         No active constraints → differences are purely geometric.
# Output: results/example2/
#
# Usage:  cd jcode && julia --project=. scripts/s30_example2.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random

logpath, tee, logfile = setup_logging("s30_example2")

# ── Configuration ───────────────────────────────────────────────────────

const α_base = 0.8
const λ = 1.0
const T_final = 50.0
const save_dt = 0.05
const tol = 1e-6
const xstar = [0.4, -0.3]   # interior solution by construction

const METRICS = [:identity, :Qinv, :diag_inv]
const METRIC_LABELS = Dict(
    :identity => "Euclidean",
    :Qinv     => "M=Q^{-1}",
    :diag_inv => "M=diag_inv",
)

const INIT_POINTS = [
    [0.9, 0.9], [-0.9, 0.9], [0.9, -0.9], [-0.9, -0.9],
    [0.9, 0.0], [-0.9, 0.0], [0.0, 0.9], [0.0, -0.9]
]

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "example2")
mkpath(results_dir)

# ── Main experiment ─────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Example 2: 2D Nonlinear QVI — Interior Solution")
@printf(tee, "  κ=25, γ=0.3, c=2.0, δ=0.10, α_base=%.2f, T=%.0f\n", α_base, T_final)
@printf(tee, "  x̄ = %s\n", string(xstar))
println(tee, "  Metrics: ", join([METRIC_LABELS[m] for m in METRICS], ", "))
println(tee, "=" ^ 70)

summary_lines = String[]
push!(summary_lines, "metric,x0_idx,x0,alpha,t_tol_2,t_tol_6,r_final,V_final,iters")

for met in METRICS
    prob_tmp = get_problem(3; metric=met)
    α_met = α_base / opnorm(prob_tmp.M)
    cfg = SolverConfig(T=T_final, alpha=α_met, lambda=λ, tol=tol)
    println(tee, "\n─── Metric: $(METRIC_LABELS[met])  α=$(round(α_met, sigdigits=4)) ───")

    for (idx, x0) in enumerate(INIT_POINTS)
        prob = get_problem(3; metric=met)
        prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                              M=prob.M, x0=x0, n=prob.n, name=prob.name)

        ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=save_dt, xstar=xstar)

        # Save trajectory
        traj_file = joinpath(results_dir, "traj_$(met)_x0_$(idx).csv")
        open(traj_file, "w") do io
            println(io, "t,x1,x2,residual,V")
            for k in eachindex(ts)
                @printf(io, "%.6f,%.10e,%.10e,%.10e,%.10e\n",
                        ts[k], xs[k][1], xs[k][2], rs[k], Vs[k])
            end
        end

        t2 = time_to_tol(ts, rs, 1e-2)
        t6 = time_to_tol(ts, rs, 1e-6)
        r_final = rs[end]
        V_final = Vs[end]

        push!(summary_lines, @sprintf("%s,%d,[%.1f;%.1f],%.4f,%.4f,%.4f,%.4e,%.4e,%d",
              met, idx, x0[1], x0[2], α_met, t2, t6, r_final, V_final, length(ts)))

        @printf(tee, "  x0 #%d [%5.1f,%5.1f]  r=%.2e  V=%.2e  t(1e-2)=%5.1f  t(1e-6)=%5.1f\n",
                idx, x0[1], x0[2], r_final, V_final, t2, t6)
    end
end

# Save summary
open(joinpath(results_dir, "summary.csv"), "w") do io
    for line in summary_lines; println(io, line); end
end

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $results_dir")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
