# ============================================================================
# s36: OAT Sweep — Effect of Step-Size α
# ============================================================================
#
# Goal:   Map the α regime: where does convergence hold, where does it break?
#         Compare theoretical stability boundary with empirical one.
# Output: results/alpha_sweep/
#
# Usage:  cd jcode && julia --project=. scripts/s36_stepsize_sweep.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf

logpath, tee, logfile = setup_logging("s36_stepsize_sweep")

# ── Configuration ───────────────────────────────────────────────────────

const κ = 50.0
const ϕ = π / 6
const δ = 0.1
const λ = 1.0
const T_final = 30.0
const tol = 1e-6
const x0 = [0.9, -0.8]
const xstar = [0.5, 0.3]

# α values: logarithmic sweep from very small to very large
const ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]

const METRICS = [:identity, :Qinv]
const METRIC_LABELS = Dict(:identity => "Euclidean", :Qinv => "M=Q^-1")

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "alpha_sweep")
mkpath(results_dir)

# ── Main sweep ──────────────────────────────────────────────────────────

println(tee, "=" ^ 70)
println(tee, "Step-Size Sweep: α effect on convergence")
@printf(tee, "  κ=%.0f, δ=%.2f, T=%.0f (adaptive stepping)\n", κ, δ, T_final)
println(tee, "  α values: ", ALPHAS)
println(tee, "  Metrics: ", join(values(METRIC_LABELS), ", "))
println(tee, "=" ^ 70)

for met in METRICS
    prob_tmp = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
    M_norm = opnorm(prob_tmp.M)

    println(tee, "\n─── Metric: $(METRIC_LABELS[met])  ‖M‖=$(round(M_norm, sigdigits=4)) ───")
    @printf(tee, "  %-10s %-8s %-12s %-8s %-8s %-8s\n",
            "α", "α‖M‖K", "r_final", "t(1e-2)", "t(1e-6)", "status")
    println(tee, "  " * "-" ^ 58)

    csv_path = joinpath(results_dir, "sweep_$(met).csv")
    open(csv_path, "w") do io
        println(io, "alpha,alpha_M_K,r_final,V_final,t_tol2,t_tol6,converged,diverged")

        for α in ALPHAS
            cfg = SolverConfig(T=T_final, alpha=α, lambda=λ, tol=tol)

            prob = get_problem(1; κ=κ, ϕ=ϕ, δ=δ, metric=met)
            prob_x0 = QVIProblem(F=prob.F, m=prob.m, proj_S=prob.proj_S,
                                  M=prob.M, x0=x0, n=prob.n, name=prob.name)

            ts, xs, rs, Vs = solve_qvi_diffeq(prob_x0, cfg; save_dt=1.0, xstar=xstar)

            t2 = time_to_tol(ts, rs, 1e-2)
            t6 = time_to_tol(ts, rs, 1e-6)
            r_final = rs[end]
            V_final = Vs[end]
            conv = r_final < tol
            divg = any(isnan, xs[end]) || norm(xs[end]) > 1e10
            α_MK = α * M_norm * κ  # effective step: α·‖M‖·K

            status = divg ? "DIVERGE" : (conv ? "OK" : (r_final < 1e-2 ? "~" : "SLOW"))

            @printf(io, "%.6f,%.4f,%.6e,%.6e,%.4f,%.4f,%s,%s\n",
                    α, α_MK, r_final, V_final, t2, t6, conv, divg)
            flush(io)

            @printf(tee, "  %-10.4f %-8.2f %-12.2e %-8.1f %-8.1f %-8s\n",
                    α, α_MK, r_final, t2, t6, status)
        end
    end
end

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $results_dir")
println(tee, "Theory predicts: α·L·K + K_m < 1 for Euclidean nonexpansiveness.")
println(tee, "For M=I, L=1, K=50: α < (1-0.1)/50 = 0.018")
println(tee, "For M=Q⁻¹, L=1, K=50: α < 0.018 (same since ‖M⁻¹‖·K is what matters)")
println(tee, "Empirical boundary should be much more permissive.")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
