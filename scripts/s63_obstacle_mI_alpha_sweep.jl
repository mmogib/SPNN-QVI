# ============================================================================
# s63: M = I step-size sweep on the obstacle QVI (fairness check, R2.3/B6)
# ============================================================================
#
# Goal:   Assess whether the failure of the Euclidean configuration M = I on
#         the obstacle problem (Experiment 5, n=20) at the protocol step size
#         is a tuning artifact: sweep alpha over five decades (including the
#         protocol value) and record, per alpha, the all-start success under
#         the common reference residual and the worst residual/hit time. The
#         data decide; runs that stop at the step budget before the horizon
#         are censored, not evidence of a plateau.
#
# Output: results/obstacle_mI_sweep/{raw.csv, summary.csv}
#         results/logs/s63_obstacle_mI_alpha_sweep_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s63_obstacle_mI_alpha_sweep.jl
#   julia --project=. scripts/s63_obstacle_mI_alpha_sweep.jl --quick
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics

function main()
    QUICK = "--quick" in ARGS

    logpath, tee, logfile = setup_logging("s63_obstacle_mI_alpha_sweep")

    # ── Configuration (matches Experiment 5: n=20, dense implementation) ────
    n_grid  = 20
    δ       = 0.1
    λ       = 1.0
    T_FINAL = 200.0                 # longer horizon than s60 (T=100), to be safe
    tol     = 1e-6
    save_dt = 0.5
    # Grid includes the exact protocol value alpha = 0.8 (codex 005 §2)
    ALPHAS  = sort(vcat(QUICK ? (10.0 .^ range(-4, 1, length=11)) : (10.0 .^ range(-4, 1, length=21)), [0.8]))

    h = 1.0 / (n_grid + 1)
    grid = [i * h for i in 1:n_grid]
    ψ0 = 0.2 * sin.(π * grid)
    K_A = (4.0 / h^2) * cos(π * h / 2)^2      # λmax(A), for the reference residual
    τ = 1.0 / K_A

    STARTS = [0.5 * (ψ0 .+ 1.0), ψ0 .+ 0.01, ones(n_grid), 0.5 * ones(n_grid)]
    START_LABELS = ["midway", "near_obstacle", "flat_1", "flat_0.5"]

    println(tee, "=" ^ 74)
    println(tee, "s63: M = I alpha sweep on the obstacle QVI (n=$(n_grid), T=$(T_FINAL))")
    @printf(tee, "  alpha grid: %d points in [1e-4, 1e1];  tau_ref = %.3e\n", length(ALPHAS), τ)
    println(tee, "=" ^ 74)

    results_dir = joinpath(@__DIR__, "..", "results", "obstacle_mI_sweep")
    mkpath(results_dir)

    # Stopping: R_ref only, evaluated at accepted endpoints; native rule OFF.
    raw_lines = String[]
    push!(raw_lines, "alpha,start,R_ref_final,r_native_final,hit_Rref,t_hit,t_end,naccept,nreject,retcode")

    all_hit_by_alpha = Dict{Float64,Bool}()
    worst_by_alpha = Dict{Float64,Float64}()
    thit_by_alpha = Dict{Float64,Float64}()

    for α in ALPHAS
        Rfinals = Float64[]; hits = Bool[]; thits = Float64[]
        for (sidx, x0) in enumerate(STARTS)
            prob0 = get_problem(5; n=n_grid, δ=δ, metric=:identity)
            prob = with_x0(prob0, x0)
            cfg = SolverConfig(T=T_FINAL, alpha=α, lambda=λ, tol=0.0)
            stop_fn = (u, t) -> reference_residual_lb(u, prob, τ, ψ0) <= tol
            ts, xs, rs, _, sol = solve_qvi_diffeq(prob, cfg; save_dt=save_dt,
                terminate_on_tol=false, stop_fn=stop_fn, compute_rs=false, return_sol=true)
            R_ref = reference_residual_lb(xs[end], prob, τ, ψ0)
            r_fin = residual(xs[end], prob, cfg)[2]
            hit = R_ref <= tol
            t_end = ts[end]
            t_hit = hit ? t_end : Inf
            push!(Rfinals, R_ref); push!(hits, hit); push!(thits, t_hit)
            push!(raw_lines, @sprintf("%.6e,%s,%.4e,%.4e,%d,%.4f,%.4f,%d,%d,%s",
                  α, START_LABELS[sidx], R_ref, r_fin, hit, t_hit, t_end,
                  sol.stats.naccept, sol.stats.nreject, string(sol.retcode)))
        end
        all_hit_by_alpha[α] = all(hits)
        worst_by_alpha[α] = maximum(Rfinals)
        thit_by_alpha[α] = all(hits) ? maximum(thits) : Inf
        @printf(tee, "  alpha=%.3e   all-start hit=%d   worst R_ref=%.3e   worst t_hit=%.1f\n",
                α, all(hits), maximum(Rfinals), all(hits) ? maximum(thits) : Inf)
    end

    # ── Summary (decision statistic: ALL-START success + worst residual) ────
    summary_lines = String[]
    push!(summary_lines, "alpha,all_start_hit,worst_R_ref,worst_t_hit")
    for α in sort(collect(keys(worst_by_alpha)))
        push!(summary_lines, @sprintf("%.6e,%d,%.4e,%.4f",
              α, all_hit_by_alpha[α], worst_by_alpha[α], thit_by_alpha[α]))
    end
    good = sort([α for α in keys(all_hit_by_alpha) if all_hit_by_alpha[α]])
    push!(summary_lines, isempty(good) ?
        "# conclusion: no tested alpha reaches the reference threshold from all starts" :
        @sprintf("# conclusion: all-start R_ref success at tested points: %s", join(good, " ")))

    suffix = QUICK ? "_quick" : ""
    open(joinpath(results_dir, "raw$(suffix).csv"), "w") do io
        foreach(l -> println(io, l), raw_lines)
    end
    open(joinpath(results_dir, "summary$(suffix).csv"), "w") do io
        foreach(l -> println(io, l), summary_lines)
    end

    println(tee, isempty(good) ? "\nNo tested alpha reaches R_ref <= $(tol) from all starts." :
            "\nAll-start R_ref success at: $(join(good, ", "))")
    println(tee, "Results saved to: $results_dir")
    teardown_logging(tee, logpath)
end

main()
