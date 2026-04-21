# ============================================================================
# s10: Smoke Test for SPNNQVI
# ============================================================================
#
# Goal:   Verify module loads, problems work, solver runs, projection correct.
# Usage:  cd jcode && julia --project=. scripts/s10_smoke_test.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf

logpath, tee, logfile = setup_logging("s10_smoke_test")

println(tee, "=" ^ 60)
println(tee, "SPNNQVI Smoke Test")
println(tee, "=" ^ 60)

# ── 1. Load problems ────────────────────────────────────────────────────
println(tee, "\n[1] Problem loading:")
for met in [:identity, :Q, :diag]
    prob = get_problem(1; metric=met)
    Fx = prob.F(prob.x0)
    mx = prob.m(prob.x0)
    @printf(tee, "  Problem 1 (%-10s): n=%d, ‖F(x0)‖=%.4e, ‖m(x0)‖=%.4e, cond(M)=%.1f\n",
            met, prob.n, norm(Fx), norm(mx), cond(prob.M))
end

# ── 2. Projection test ──────────────────────────────────────────────────
println(tee, "\n[2] Metric projection (identity metric):")
prob_I = get_problem(1; metric=:identity)
z_interior = [0.3, 0.2]
z_exterior = [1.5, -1.5]
p_int = metric_projection(z_interior, prob_I)
p_ext = metric_projection(z_exterior, prob_I)
@printf(tee, "  Interior [0.3, 0.2] → %s (should stay)\n", string(round.(p_int, digits=4)))
@printf(tee, "  Exterior [1.5,-1.5] → %s (should clip)\n", string(round.(p_ext, digits=4)))

println(tee, "\n[3] Metric projection (M=Q, non-identity):")
prob_Q = get_problem(1; metric=:Q)
p_Q = metric_projection(z_exterior, prob_Q)
@printf(tee, "  Exterior [1.5,-1.5] → %s (metric-projected)\n", string(round.(p_Q, digits=4)))

# ── 3. T-map and residual ──────────────────────────────────────────────
println(tee, "\n[4] T-map and residual:")
cfg = SolverConfig(alpha=0.1, lambda=1.0, dt=0.01, T=1.0, tol=1e-6)
for met in [:identity, :Q]
    prob = get_problem(1; metric=met)
    Tx = T_map(prob.x0, prob, cfg)
    r, rn = residual(prob.x0, prob, cfg)
    @printf(tee, "  %-10s: T(x0)=%s, ‖r(x0)‖=%.4e\n",
            met, string(round.(Tx, digits=4)), rn)
end

# ── 4. Short solver run ─────────────────────────────────────────────────
println(tee, "\n[5] Solver (short run, 1000 steps, α scaled by 1/‖M‖):")
for met in [:identity, :Q]
    prob = get_problem(1; metric=met)
    α_scaled = 0.1 / opnorm(prob.M)
    cfg_short = SolverConfig(alpha=α_scaled, lambda=1.0, dt=0.01, T=10.0, tol=1e-6, maxiter=1000)
    result = solve_qvi(prob, cfg_short)
    @printf(tee, "  %-10s (α=%.4f): %6s after %4d iters, r_final=%.4e, time=%.3fs\n",
            met, α_scaled, result.status, result.iterations, result.residual_final, result.time_seconds)
end

# ── 5. ODE trajectory ──────────────────────────────────────────────────
println(tee, "\n[6] ODE trajectory (T=5, save_dt=1.0, M=Q):")
prob = get_problem(1; metric=:Q)
α_scaled = 0.1 / opnorm(prob.M)
cfg_ode = SolverConfig(alpha=α_scaled, lambda=1.0, dt=0.01, T=5.0, tol=1e-6)
xstar = [0.5, 0.3]
ts, xs, rs, Vs = solve_qvi_ode(prob, cfg_ode; save_dt=1.0, xstar=xstar)
for k in eachindex(ts)
    @printf(tee, "  t=%.1f: x=%s, ‖r‖=%.4e, V=%.4e\n",
            ts[k], string(round.(xs[k], digits=4)), rs[k], Vs[k])
end

# ── 7. DiffEq solver (adaptive stepping) ─────────────────────────────
println(tee, "\n[7] DiffEq solver (Tsit5, adaptive, M=Q):")
prob = get_problem(1; metric=:Q)
α_scaled = 0.1 / opnorm(prob.M)
cfg_de = SolverConfig(alpha=α_scaled, lambda=1.0, T=5.0, tol=1e-6)
ts_de, xs_de, rs_de, Vs_de = solve_qvi_diffeq(prob, cfg_de; save_dt=1.0, xstar=xstar)
for k in eachindex(ts_de)
    @printf(tee, "  t=%.1f: x=%s, ‖r‖=%.4e, V=%.4e\n",
            ts_de[k], string(round.(xs_de[k], digits=4)), rs_de[k], Vs_de[k])
end

println(tee, "\n" * "=" ^ 60)
println(tee, "Smoke test complete.")
println(tee, "=" ^ 60)

teardown_logging(tee, logpath)
