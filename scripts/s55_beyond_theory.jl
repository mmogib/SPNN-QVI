# ============================================================================
# s55: Beyond Theory — Pushing the Limits
# ============================================================================
#
# Goal:   Test convergence beyond the proven regime. Explore conjectures.
#         Tests: (1) K_m > 1 with weak monotonicity
#                (2) Merely monotone F (not strongly)
#                (3) Very large α
#                (4) Non-monotone F
# Output: results/beyond_theory/
#
# Usage:  cd jcode && julia --project=. scripts/s55_beyond_theory.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using OrdinaryDiffEq
using LinearAlgebra, Printf

logpath, tee, logfile = setup_logging("s55_beyond_theory")

results_dir = joinpath(@__DIR__, "..", "results", "beyond_theory")
mkpath(results_dir)

println(tee, "=" ^ 70)
println(tee, "Beyond Theory: Pushing the Limits of SPNN-QVI")
println(tee, "=" ^ 70)

# Helper: run one test and report
function run_test(tee, name, prob, cfg, xstar; solver=Tsit5())
    ts, xs, rs, Vs = solve_qvi_diffeq(prob, cfg; solver=solver, save_dt=1.0, xstar=xstar)
    r_final = rs[end]
    V_final = Vs[end]
    divg = any(isnan, xs[end]) || norm(xs[end]) > 1e10
    t2 = time_to_tol(ts, rs, 1e-2)
    t6 = time_to_tol(ts, rs, 1e-6)
    status = divg ? "DIVERGE" : (r_final < 1e-6 ? "CONVERGE" : (r_final < 1e-2 ? "SLOW" : "STALL"))
    @printf(tee, "  %-40s  r=%.2e  V=%.2e  t2=%5.1f  t6=%5.1f  %s\n",
            name, r_final, V_final, t2, t6, status)
    return (name=name, r_final=r_final, V_final=V_final, t2=t2, t6=t6, status=status, divg=divg)
end

# Common setup
rot2(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Large K_m with WEAK monotonicity (small μ)
# ═══════════════════════════════════════════════════════════════════════

println(tee, "\n━━━ TEST 1: Large K_m with weak monotonicity ━━━")
println(tee, "  F(x) = Qx - b with Q nearly singular (μ_min = 0.1)")
println(tee, "  Sweep K_m = δ from 0 to 2.0")

begin
    n = 2
    R = rot2(π/6)
    Q_weak = Symmetric(R' * Diagonal([5.0, 0.1]) * R)  # weak: μ=0.1, K=5
    xbar = [0.5, 0.3]
    b_weak = Q_weak * xbar
    M_weak = Matrix{Float64}(inv(Q_weak))
    lb, ub = -ones(n), ones(n)
    x0 = [0.9, -0.8]

    csv_io = open(joinpath(results_dir, "test1_km_weak.csv"), "w")
    println(csv_io, "delta,r_final,V_final,t2,t6,status")

    for δ in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0, 1.2, 1.5, 2.0]
        F(x) = Q_weak * x - b_weak
        m(x) = δ * x
        proj_S(z) = clamp.(z, lb, ub)
        prob = QVIProblem(F=F, m=m, proj_S=proj_S, M=M_weak, x0=x0, n=n, name="weak_km_$δ")
        α = 0.8 / opnorm(M_weak)
        cfg = SolverConfig(T=100.0, alpha=α, lambda=1.0, tol=1e-6)
        res = run_test(tee, @sprintf("δ=%.2f (K_m=%.2f, μ=0.1)", δ, δ), prob, cfg, xbar)
        @printf(csv_io, "%.4f,%.6e,%.6e,%.4f,%.4f,%s\n",
                δ, res.r_final, res.V_final, res.t2, res.t6, res.status)
        flush(csv_io)
    end
    close(csv_io)
end

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Merely monotone F (μ = 0, not strongly monotone)
# ═══════════════════════════════════════════════════════════════════════

println(tee, "\n━━━ TEST 2: Merely monotone F (μ = 0) ━━━")
println(tee, "  F(x) = [x₁; 0] — monotone but NOT strongly monotone")
println(tee, "  Solution set is a line, not a point")

begin
    n = 2
    # F(x) = [x1; 0] is monotone (⟨F(x)-F(y), x-y⟩ = (x1-y1)² ≥ 0) but not strongly
    # QVI on [-1,1]² with m(x) = δx: solution is any x with x1=0 and x ∈ δx + [-1,1]²
    F_mono(x) = [x[1]; 0.0]
    lb, ub = -ones(2), ones(2)
    proj_S(z) = clamp.(z, lb, ub)
    M_I = Matrix{Float64}(I, 2, 2)

    csv_io = open(joinpath(results_dir, "test2_merely_monotone.csv"), "w")
    println(csv_io, "delta,x0,r_final,x_final_1,x_final_2,status")

    for δ in [0.0, 0.1, 0.5]
        for x0 in [[0.8, 0.7], [-0.5, 0.3], [0.2, -0.9]]
            m(x) = δ * x
            prob = QVIProblem(F=F_mono, m=m, proj_S=proj_S, M=M_I, x0=x0, n=2, name="mono_δ$(δ)")
            cfg = SolverConfig(T=50.0, alpha=0.8, lambda=1.0, tol=1e-6)
            ts, xs, rs, Vs = solve_qvi_diffeq(prob, cfg; save_dt=1.0, xstar=nothing)
            r_final = rs[end]
            xf = xs[end]
            status = r_final < 1e-6 ? "CONVERGE" : (r_final < 1e-2 ? "SLOW" : "STALL")
            @printf(tee, "  δ=%.1f x0=[%5.1f,%5.1f]  r=%.2e  x_final=[%6.3f,%6.3f]  %s\n",
                    δ, x0[1], x0[2], r_final, xf[1], xf[2], status)
            @printf(csv_io, "%.2f,[%.1f;%.1f],%.6e,%.6f,%.6f,%s\n",
                    δ, x0[1], x0[2], r_final, xf[1], xf[2], status)
        end
    end
    close(csv_io)
end

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Non-monotone (skew-symmetric) F
# ═══════════════════════════════════════════════════════════════════════

println(tee, "\n━━━ TEST 3: Non-monotone F (skew-symmetric + small diagonal) ━━━")
println(tee, "  F(x) = (εI + J)x - b where J is skew-symmetric")
println(tee, "  ε controls how far from monotone")

begin
    n = 2
    lb, ub = -ones(n), ones(n)
    proj_S(z) = clamp.(z, lb, ub)
    M_I = Matrix{Float64}(I, n, n)
    xbar = [0.5, 0.3]
    J = [0.0 -1.0; 1.0 0.0]   # skew-symmetric

    csv_io = open(joinpath(results_dir, "test3_nonmonotone.csv"), "w")
    println(csv_io, "epsilon,delta,r_final,V_final,status")

    for ε in [1.0, 0.5, 0.1, 0.01, 0.0, -0.1]
        A = ε * I(2) + J
        b_nm = A * xbar
        F_nm(x) = A * x - b_nm
        for δ in [0.0, 0.1, 0.5]
            m(x) = δ * x
            prob = QVIProblem(F=F_nm, m=m, proj_S=proj_S, M=M_I, x0=[0.9, -0.8], n=n,
                              name="nonmono_ε$(ε)_δ$(δ)")
            cfg = SolverConfig(T=50.0, alpha=0.5, lambda=1.0, tol=1e-6)
            res = run_test(tee, @sprintf("ε=%5.2f δ=%.1f", ε, δ), prob, cfg, xbar)
            @printf(csv_io, "%.4f,%.4f,%.6e,%.6e,%s\n",
                    ε, δ, res.r_final, res.V_final, res.status)
            flush(csv_io)
        end
    end
    close(csv_io)
end

# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Very large α (does projection prevent divergence?)
# ═══════════════════════════════════════════════════════════════════════

println(tee, "\n━━━ TEST 4: Extreme α values ━━━")
println(tee, "  Affine QVI with strong Q (κ=50), test α up to 1000")

begin
    n = 2
    R = rot2(π/6)
    Q = Symmetric(R' * Diagonal([50.0, 1.0]) * R)
    xbar = [0.5, 0.3]
    b = Q * xbar
    lb, ub = -ones(n), ones(n)
    proj_S(z) = clamp.(z, lb, ub)
    M_I = Matrix{Float64}(I, n, n)

    csv_io = open(joinpath(results_dir, "test4_extreme_alpha.csv"), "w")
    println(csv_io, "alpha,delta,r_final,V_final,status")

    for α in [10.0, 50.0, 100.0, 500.0, 1000.0]
        for δ in [0.0, 0.1, 0.5]
            F(x) = Q * x - b
            m(x) = δ * x
            prob = QVIProblem(F=F, m=m, proj_S=proj_S, M=M_I, x0=[0.9, -0.8], n=n,
                              name="extreme_α$(α)_δ$(δ)")
            cfg = SolverConfig(T=30.0, alpha=α, lambda=1.0, tol=1e-6)
            res = run_test(tee, @sprintf("α=%7.1f δ=%.1f", α, δ), prob, cfg, xbar;
                           solver=AutoTsit5(Rosenbrock23()))
            @printf(csv_io, "%.4f,%.4f,%.6e,%.6e,%s\n",
                    α, δ, res.r_final, res.V_final, res.status)
            flush(csv_io)
        end
    end
    close(csv_io)
end

# ═══════════════════════════════════════════════════════════════════════

println(tee, "\n" * "=" ^ 70)
println(tee, "Results saved to: $results_dir")
println(tee, "=" ^ 70)

teardown_logging(tee, logpath)
