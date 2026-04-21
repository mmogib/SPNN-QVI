# ============================================================================
# s35: OAT Sweep — Effect of Contraction Constant K_m
# ============================================================================
#
# Goal:   Study how K_m (= δ) affects convergence rate and stability.
#         Tests theory: convergence requires K_m < 1.
# Output: results/km_sweep/sweep.csv
#         results/logs/s35_contraction_sweep_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s35_contraction_sweep.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf

# ── Logging ─────────────────────────────────────────────────────────────

logpath, tee, logfile = setup_logging("s35_contraction_sweep")

# ── Configuration ───────────────────────────────────────────────────────

const κ = 50.0
const ϕ = π / 6
const α_base = 0.8
const λ = 1.0
const T_final = 50.0
const tol = 1e-6
const x0 = [0.9, -0.8]
# NOTE: xstar depends on δ since the QVI solution shifts with m.
# We compute it per δ. For δ=0, xstar=[0.5,0.3] by construction (F(xstar)=0).
# For δ>0, the solution may differ — we track convergence via residual only.

const DELTAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.1, 1.5]
const METRIC = :Qinv

# ── Output setup ────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "..", "results", "km_sweep")
mkpath(results_dir)

# ── Main sweep ──────────────────────────────────────────────────────────

prob_ref = get_problem(1; κ=κ, ϕ=ϕ, δ=0.1, metric=METRIC)
const α = α_base / opnorm(prob_ref.M)

println(tee, "=" ^ 70)
println(tee, "Contraction Sweep: K_m effect on convergence")
@printf(tee, "  Metric: M=Q⁻¹, α=%.4f (α_base=%.2f), T=%.0f\n", α, α_base, T_final)
println(tee, "  δ values: ", DELTAS)
println(tee, "=" ^ 70)

cfg = SolverConfig(T=T_final, alpha=α, lambda=λ, tol=tol)

csv_path = joinpath(results_dir, "sweep.csv")
open(csv_path, "w") do io
    println(io, "delta,r_final,t_tol2,t_tol6,converged,iters")

    # Build a problem with boundary solution: F(x) = Qx + q with q chosen
    # so that the VI solution on [-1,1]² hits the boundary.
    R = [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]
    Q = Symmetric(R' * Diagonal([κ, 1.0]) * R)
    q_boundary = [5.0, -3.0]   # pushes solution toward boundary
    Minv = inv(Matrix(Q))

    for δ in DELTAS
        F(x) = Q * x + q_boundary
        m(x) = δ * x
        lb, ub = -ones(2), ones(2)
        proj_S(z) = clamp.(z, lb, ub)
        M_mat = Matrix{Float64}(Minv)   # M = Q⁻¹

        prob = QVIProblem(F=F, m=m, proj_S=proj_S, M=M_mat, x0=x0, n=2,
                          name="boundary_delta_$(δ)")

        xstar_approx = nothing  # unknown; track residual only
        ts, xs, rs, Vs = solve_qvi_diffeq(prob, cfg; save_dt=0.5, xstar=xstar_approx)

        t2 = time_to_tol(ts, rs, 1e-2)
        t6 = time_to_tol(ts, rs, 1e-6)
        r_final = rs[end]
        conv = r_final < tol

        @printf(io, "%.4f,%.6e,%.4f,%.4f,%s,%d\n",
                δ, r_final, t2, t6, conv, length(ts))
        flush(io)

        status_str = conv ? "OK" : (r_final < 1e-2 ? "~" : "FAIL")
        @printf(tee, "  δ=%.4f  K_m=%.4f  r_final=%.2e  t(1e-2)=%6.1f  t(1e-6)=%6.1f  %s\n",
                δ, δ, r_final, t2, t6, status_str)
    end
end

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $csv_path")
println(tee, "Key question: Does convergence persist as K_m -> 1? Beyond K_m = 1?")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
