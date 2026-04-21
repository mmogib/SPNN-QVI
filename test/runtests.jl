# Test suite for SPNNQVI
#
# Run: julia --project=. test/runtests.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using Test, LinearAlgebra

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

    # ── Projection ──────────────────────────────────────────────────────────
    @testset "Euclidean projection (identity metric)" begin
        proj_S(z) = clamp.(z, 0.0, 1.0)
        M = Matrix{Float64}(I, 2, 2)
        # Interior point: should not move
        z1 = [0.5, 0.3]
        @test metric_projection(z1, M, proj_S) ≈ z1
        # Exterior point: should project onto box
        z2 = [1.5, -0.3]
        @test metric_projection(z2, M, proj_S) ≈ [1.0, 0.0]
    end

    @testset "Translated projection identity" begin
        proj_S(z) = clamp.(z, 0.0, 1.0)
        M = Matrix{Float64}(I, 2, 2)
        mx = [0.1, 0.2]
        z = [0.8, 0.9]
        # P_{mx+S,M}(z) should equal mx + P_{S,M}(z - mx)
        result = metric_projection_translated(z, mx, M, proj_S)
        expected = mx + metric_projection(z - mx, M, proj_S)
        @test result ≈ expected
    end

    # ── Residual ────────────────────────────────────────────────────────────
    @testset "Residual computation" begin
        prob = get_problem(1)
        r = compute_residual(prob.x0, prob, 0.1)
        @test length(r) == prob.n
        @test all(isfinite.(r))
    end

    # ── I/O Utilities ───────────────────────────────────────────────────────
    @testset "TeeIO" begin
        buf = IOBuffer()
        tee = TeeIO(devnull, buf)
        print(tee, "hello")
        flush(tee)
        @test String(take!(buf)) == "hello"
    end

end  # @testset "SPNNQVI"
