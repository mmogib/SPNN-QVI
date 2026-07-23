# Test suite for SPNNQVI
#
# Run: julia --project=. test/runtests.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using Test, LinearAlgebra, SparseArrays, Random

@testset "SPNNQVI" begin

    # ── Types & Config ──────────────────────────────────────────────────────
    @testset "Config defaults" begin
        cfg = SolverConfig()
        @test cfg.tol == 1e-6
        @test cfg.maxiter == 100_000
        @test cfg.verbose == false
        @test cfg.alpha == 0.1
        @test cfg.lambda == 1.0
    end

    # ── Problem Loading ─────────────────────────────────────────────────────
    @testset "Problem loading" begin
        for id in [1, 2]
            prob = get_problem(id)
            @test prob.n > 0
            Fx = prob.F(prob.x0)
            @test length(Fx) == prob.n
            @test all(isfinite.(Fx))
            mx = prob.m(prob.x0)
            @test length(mx) == prob.n
            @test all(isfinite.(mx))
            @test size(prob.M) == (prob.n, prob.n)
            @test isposdef(prob.M)
        end
    end

    # ── Projection (current 2-arg problem-based API) ────────────────────────
    @testset "Euclidean projection (identity metric)" begin
        prob = get_problem(1; metric=:identity)   # S = [-1,1]^2, M = I
        # Interior point: should not move
        z1 = [0.5, 0.3]
        @test metric_projection(z1, prob) ≈ z1
        # Exterior point: should project onto box
        z2 = [1.5, -1.3]
        @test metric_projection(z2, prob) ≈ [1.0, -1.0]
    end

    @testset "Translated projection identity" begin
        prob = get_problem(1; metric=:Qinv)
        mx = prob.m(prob.x0)
        z = [0.8, 0.9]
        # P_{mx+S,M⁻¹}(z) should equal mx + P_{S,M⁻¹}(z - mx)
        result = metric_projection_translated(z, mx, prob)
        expected = mx + metric_projection(z - mx, prob)
        @test result ≈ expected
    end

    @testset "Metric projection variational characterization" begin
        # (p - z)ᵀ M⁻¹ (y - p) ≥ 0 for all y ∈ S at p = P_{S,M⁻¹}(z)
        prob = get_problem(1; metric=:Qinv)
        Minv = inv(prob.M)
        z = [1.7, -2.3]
        p = metric_projection(z, prob)
        rng = MersenneTwister(7)
        for _ in 1:50
            y = clamp.(2 .* rand(rng, 2) .- 1, -1.0, 1.0)
            @test dot(p - z, Minv * (y - p)) >= -1e-8
        end
    end

    # ── Refactor: fast paths agree with the legacy dense path ───────────────
    @testset "Diagonal-metric fast path = dense coordinate descent" begin
        n = 8
        rng = MersenneTwister(11)
        d = 0.5 .+ rand(rng, n)
        lb, ub = -ones(n), ones(n)
        for _ in 1:20
            z = 4 .* randn(rng, n)
            # Dense CD on the diagonal QP vs. plain clamp (the fast path)
            y_cd = metric_project_box_cholesky(z, Matrix(Diagonal(d)), lb, ub)
            @test y_cd ≈ clamp.(z, lb, ub) atol=1e-10
        end
    end

    @testset "Sparse CD = dense CD (tridiagonal metric)" begin
        n = 30
        h = 1.0 / (n + 1)
        A_sp = spdiagm(-1 => fill(-1.0/h^2, n-1), 0 => fill(2.0/h^2, n), 1 => fill(-1.0/h^2, n-1))
        A_dense = Matrix(A_sp)
        lb, ub = zeros(n), fill(Inf, n)
        rng = MersenneTwister(13)
        for _ in 1:10
            z = randn(rng, n)
            y_sp = SPNNQVI._project_box_cd_sparse(z, A_sp, lb, ub)
            y_de = metric_project_box_cholesky(z, A_dense, lb, ub)
            @test y_sp ≈ y_de atol=1e-8
        end
    end

    @testset "PDAS lower-bound projection (verified KKT, fail-closed)" begin
        for n in [15, 200, 800]
            h = 1.0 / (n + 1)
            Q = spdiagm(-1 => fill(-1.0/h^2, n-1), 0 => fill(2.0/h^2, n), 1 => fill(-1.0/h^2, n-1))
            grid = [i*h for i in 1:n]
            lb = 0.2 * sin.(π * grid)
            sQ = 2.0 / h^2                              # max |Q| entry (scale)
            rng = MersenneTwister(23)
            for _ in 1:5
                z = lb .+ 0.3 .* randn(rng, n)          # mixed active/inactive
                out = SPNNQVI._project_lb_pdas_info(z, Q, lb)
                # Termination metadata: normal convergence, well below the cap
                @test out.converged
                @test out.reason in (:stable, :cycle)
                @test out.iters < 50
                @test out.kkt_ok
                # Independent scale-aware KKT verification (recomputed here,
                # not trusting the routine's own check)
                y = out.y
                μ = Q * (y - z)
                zs = max(1.0, maximum(abs, z), maximum(abs, lb))
                @test minimum(y .- lb) >= -1e-11 * zs                     # primal
                inact = (y .- lb) .> 1e-11 * zs
                @test maximum(abs.(μ[inact]); init=0.0) <= 1e-9 * sQ * zs  # stationarity
                @test minimum(μ[.!inact]; init=0.0) >= -1e-9 * sQ * zs     # dual
                @test maximum(abs.(μ .* (y .- lb)); init=0.0) <= 1e-9 * sQ * zs^2  # compl. (componentwise)
                # Fail-closed wrapper returns the same verified point
                @test SPNNQVI._project_lb_pdas(z, Q, lb) == y
            end
            # objective no worse than long coordinate descent (small n only)
            if n == 15
                z = lb .+ 0.3 .* randn(rng, n)
                y_pdas = SPNNQVI._project_lb_pdas(z, Q, lb)
                y_cd = SPNNQVI._project_box_cd_sparse(z, Q, lb, fill(Inf, n); maxiter=20_000)
                f(y) = 0.5 * dot(y - z, Q * (y - z))
                @test f(y_pdas) <= f(y_cd) + 1e-10
                @test y_pdas ≈ y_cd atol=1e-6
            end
        end
        # Structural routing gate: PDAS accepts only symmetric Z-matrices
        Qgood = spdiagm(-1 => [-1.0], 0 => [2.0, 2.0], 1 => [-1.0])
        Qbad  = spdiagm(-1 => [0.5],  0 => [2.0, 2.0], 1 => [0.5])   # positive off-diagonal
        @test SPNNQVI._pdas_applicable(Qgood)
        @test !SPNNQVI._pdas_applicable(Qbad)
        # Degenerate / noise-band inputs — the regime where the active-set
        # classification can chatter (multiplier AND gap at the noise floor):
        # the routine must still return a KKT-verified point, never throw.
        for n in [800, 10_000]
            h = 1.0 / (n + 1)
            Q = spdiagm(-1 => fill(-1.0/h^2, n-1), 0 => fill(2.0/h^2, n), 1 => fill(-1.0/h^2, n-1))
            grid = [i*h for i in 1:n]
            lb = 0.2 * sin.(π * grid)
            rng = MersenneTwister(29)
            out1 = SPNNQVI._project_lb_pdas_info(lb .- 1.0, Q, lb)   # fully active: y* = lb
            @test out1.kkt_ok
            @test out1.converged
            @test out1.reason in (:stable, :cycle)
            @test out1.iters <= 10                                   # iteration headroom
            @test maximum(abs, out1.y .- lb) <= 1e-9
            @test SPNNQVI._project_lb_pdas(lb .- 1.0, Q, lb) == out1.y   # wrapper path
            zg = lb .+ 1e-9 .* abs.(randn(rng, n))                   # grazing inactive: y* = z
            out2 = SPNNQVI._project_lb_pdas_info(zg, Q, lb)
            @test out2.kkt_ok
            @test out2.converged
            @test out2.reason in (:stable, :cycle)
            @test out2.iters <= 10
            @test maximum(abs, out2.y .- zg) <= 1e-9
            @test SPNNQVI._project_lb_pdas(zg, Q, lb) == out2.y      # wrapper path
            zn = lb .- 1e-9 .* abs.(randn(rng, n))                   # near-grazing active
            out3 = SPNNQVI._project_lb_pdas_info(zn, Q, lb)
            @test out3.kkt_ok
            @test out3.converged
            @test out3.reason in (:stable, :cycle)
            @test out3.iters <= 25          # headroom (near-grazing peels a little)
            @test SPNNQVI._project_lb_pdas(zn, Q, lb) == out3.y      # wrapper path
        end
        # Far-from-solution input requiring extensive active-set peeling — the
        # exact first discrete-KM projection at n=500 (deep below the obstacle,
        # spatially structured). Needs more than a constant iteration cap.
        let n = 500
            h = 1.0 / (n + 1)
            Q = spdiagm(-1 => fill(-1.0/h^2, n-1), 0 => fill(2.0/h^2, n), 1 => fill(-1.0/h^2, n-1))
            grid = [i*h for i in 1:n]
            lb = 0.2 * sin.(π * grid)
            λmin = (4.0/h^2) * sin(π*h/2)^2
            w = cholesky(Symmetric(Q)) \ ones(n)
            x0 = 0.5 .* (lb .+ 1.0)
            α = 0.8 * λmin
            zp = x0 .- (0.1 * sum(x0)/n) .* ones(n) .- α .* (x0 .- w)
            # Two-sided: the OLD constant cap of 50 provably fails here...
            out50 = SPNNQVI._project_lb_pdas_info(zp, Q, lb; maxiter=50)
            @test out50.reason === :cap
            @test !out50.kkt_ok
            @test_throws ErrorException SPNNQVI._project_lb_pdas(zp, Q, lb; maxiter=50)
            # ...while the n-scaled default converges, needs more than 50
            # iterations, and passes an independently recomputed KKT check.
            out = SPNNQVI._project_lb_pdas_info(zp, Q, lb)
            @test out.converged
            @test out.iters > 50
            @test out.kkt_ok
            y = SPNNQVI._project_lb_pdas(zp, Q, lb)                  # wrapper path
            @test y == out.y
            # Independent KKT recomputation at the PRODUCTION normalization
            # (tolP = 1e-11*zs, tolD = 1e-11*sQ*zs, compl <= tolD*zs)
            μ = Q * (y - zp)
            sQ = 2.0 / h^2
            zs = max(1.0, maximum(abs, zp), maximum(abs, lb))
            @test minimum(y .- lb) >= -1e-11 * zs
            inact = (y .- lb) .> 1e-11 * zs
            @test maximum(abs.(μ[inact]); init=0.0) <= 1e-11 * sQ * zs
            @test minimum(μ[.!inact]; init=0.0) >= -1e-11 * sQ * zs
            @test maximum(abs.(μ .* (y .- lb)); init=0.0) <= 1e-11 * sQ * zs^2
        end
    end

    @testset "FactorInverse applies A⁻¹" begin
        n = 25
        h = 1.0 / (n + 1)
        A_sp = spdiagm(-1 => fill(-1.0/h^2, n-1), 0 => fill(2.0/h^2, n), 1 => fill(-1.0/h^2, n-1))
        λmin = (4.0/h^2) * sin(π*h/2)^2
        op = FactorInverse(cholesky(Symmetric(A_sp)), n, 1.0/λmin)
        Ainv = inv(Matrix(A_sp))
        v = randn(MersenneTwister(17), n)
        @test op * v ≈ Ainv * v rtol=1e-8
        @test opnorm(op) ≈ 1.0/λmin
        @test size(op) == (n, n)
    end

    @testset "Sparse obstacle = dense obstacle" begin
        n = 50
        for met in [:identity, :jacobi, :Ainv]
            p_dense  = get_problem(5; n=n, metric=met, sparse_impl=false)
            p_sparse = get_problem(5; n=n, metric=met, sparse_impl=true)
            x = p_dense.x0 .+ 0.1
            @test p_sparse.F(x) ≈ p_dense.F(x) rtol=1e-10
            @test p_sparse.m(x) ≈ p_dense.m(x)
            cfg = SolverConfig(alpha=0.01)
            @test T_map(x, p_sparse, cfg) ≈ T_map(x, p_dense, cfg) rtol=1e-6
            @test norm_M(p_sparse) ≈ opnorm(p_dense.M) rtol=1e-6
        end
    end

    # ── Residual ────────────────────────────────────────────────────────────
    @testset "Residual computation" begin
        prob = get_problem(1)
        r = compute_residual(prob.x0, prob, 0.1)
        @test length(r) == prob.n
        @test all(isfinite.(r))
    end

    @testset "Reference residual" begin
        # R_ref vanishes at the known solution of Problem 1 (F(x̄)=0, x̄-m(x̄) interior)
        prob = get_problem(1)
        xbar = [0.5, 0.3]
        @test reference_residual(xbar, prob, 0.02) < 1e-12
        # Cancellation-reduced lower-bound version agrees with the generic one
        pobs = get_problem(5; n=30, sparse_impl=true, metric=:identity)
        tau = 1.0 / ((4.0/(1/31)^2))   # ~1/λmax scale
        x = pobs.x0 .+ 0.05 .* sin.(1:30)
        @test reference_residual_lb(x, pobs, tau, pobs.lb) ≈ reference_residual(x, pobs, tau) rtol=1e-6
    end

    # ── Fixed-point solver (Proposition 2 regime) ───────────────────────────
    @testset "Fixed-point iteration converges inside the certified window" begin
        # κ=1.2, δ=0.05, M=I: μ_eff = 1 - δκ = 0.94 > 2κ√δ ≈ 0.537 → window ≈ (0.117, 1.189)
        prob = get_problem(1; κ=1.2, δ=0.05, metric=:identity)
        cfg = SolverConfig(alpha=0.5, lambda=1.0)
        xbar = [0.5, 0.3]
        stop_fn(x, k) = reference_residual(x, prob, 0.5) < 1e-10
        out = solve_qvi_fixedpoint(prob, cfg; h=1.0, stop_fn=stop_fn, maxiter=2000)
        @test out.stopped
        @test norm(out.x_final - xbar) < 1e-8
        # Half-step KM relaxation also converges
        out2 = solve_qvi_fixedpoint(prob, cfg; h=0.5, stop_fn=stop_fn, maxiter=4000)
        @test out2.stopped
    end

    # ── Copy helper ─────────────────────────────────────────────────────────
    @testset "with_x0 preserves caches" begin
        p = get_problem(5; n=200, metric=:jacobi)   # sparse_impl auto (n > 100)
        x0 = fill(0.7, 200)
        p2 = with_x0(p, x0)
        @test p2.x0 == x0
        @test p2.Minv === p.Minv
        @test p2.lb === p.lb
        @test p2.Mnorm == p.Mnorm
    end

    # ── diffeq solver options ───────────────────────────────────────────────
    @testset "solve_qvi_diffeq stop_fn and no-terminate options" begin
        prob = get_problem(1; κ=1.2, δ=0.05, metric=:identity)
        # Certified rate is λ(1-ρ_M) ≈ 0.23 per unit time — needs T ≈ 100 for 1e-8
        cfg = SolverConfig(T=120.0, alpha=0.5, lambda=1.0, tol=1e-6)
        # Custom stopping rule at accepted endpoints
        ts, xs, rs, Vs = solve_qvi_diffeq(prob, cfg; save_dt=0.1,
            stop_fn=(u, t) -> reference_residual(u, prob, 0.5) < 1e-8)
        @test norm(xs[end] - [0.5, 0.3]) < 1e-6
        @test ts[end] < 120.0                     # stopped early by the rule
        # No early termination: runs to T
        cfg2 = SolverConfig(T=10.0, alpha=0.5, lambda=1.0, tol=1e-6)
        ts2, _, _, _ = solve_qvi_diffeq(prob, cfg2; save_dt=0.5, terminate_on_tol=false)
        @test ts2[end] ≈ 10.0 atol=0.5
    end

    # ── I/O Utilities ───────────────────────────────────────────────────────
    @testset "TeeIO" begin
        buf = IOBuffer()
        tee = TeeIO(devnull, buf)
        print(tee, "hello")
        flush(tee)
        @test String(take!(buf)) == "hello"
    end

    # ── Production-script compile checks ───────────────────────────────────
    @testset "Production scripts use literal @sprintf formats" begin
        scripts_dir = joinpath(@__DIR__, "..", "scripts")
        for script in ("s61_obstacle_finegrid.jl", "s72_baselines.jl")
            parsed = Meta.parseall(read(joinpath(scripts_dir, script), String);
                                   filename=script)
            bad_formats = LineNumberNode[]
            stack = Any[parsed]
            while !isempty(stack)
                node = pop!(stack)
                node isa Expr || continue
                if node.head === :macrocall &&
                   node.args[1] === Symbol("@sprintf") &&
                   !(node.args[3] isa String)
                    push!(bad_formats, node.args[2])
                end
                append!(stack, node.args)
            end
            @test isempty(bad_formats)
        end
    end

    @testset "s62 reports the pairwise endpoint diameter" begin
        source = read(joinpath(@__DIR__, "..", "scripts", "s62_gnep.jl"), String)
        @test !occursin("norm(p - converged_pts[1])", source)
        @test occursin("norm(converged_pts[i] - converged_pts[j])", source)
    end

    @testset "s72 promotes manifests only after gates pass" begin
        source = read(joinpath(@__DIR__, "..", "scripts", "s72_baselines.jl"), String)
        promote = first(findlast("finalize_manifest!(results_dir, SUFFIX)", source))
        boundary_gate = first(findlast("isempty(boundary_locks)", source))
        sensitivity_gate = first(findlast("gate_failed", source))
        @test boundary_gate < promote
        @test sensitivity_gate < promote
    end

end  # @testset "SPNNQVI"
