# ============================================================================
# s61: Fine-grid implicit obstacle QVI — dimension/conditioning scaling (R3.4)
# ============================================================================
#
# Goal:   Extend the obstacle experiment (Problem 5) to fine grids with the
#         sparse/operator implementation: sparse stiffness matrix, O(n) matvecs,
#         Diagonal metrics via closed-form clamp, M = A⁻¹ via sparse Cholesky.
#         κ(A) ≈ 4(n+1)²/π² grows quadratically — the conditioning story.
#
# Protocol (channels/codex_to_claude/002 §5), AMENDED AFTER THE PILOT RUN of
# 2026-07-22 and locked before the production runs (documented in
# channels/claude_to_codex/006; the original protocol declared an absolute
# threshold, which the pilot showed to lie below the attainable solver floor
# at large n):
#   * On the inactive set R_ref = ||Ax - f||, whose absolute size grows with
#     ||A||, so the original fixed ABSOLUTE threshold demands state accuracy
#     below the integrator's tolerance floor at large n. PRIMARY cross-n
#     criterion is therefore the RELATIVE residual-reduction rule
#     R_rms(x) ≤ EPS_REL · R_rms(x0) — a matched-start work criterion, not a
#     claim of common physical accuracy across grids — with the absolute rule
#     R_rms ≤ EPS_ABS reported alongside. Stopping fires on the UNION of the
#     two rules; the dominant_rule column records which one set the threshold
#     (for the declared starts R_rms(x0) > 1, so the relative rule dominates).
#   * For the 1-D Laplacian, diag(A) is CONSTANT, so the Jacobi metric is a
#     scalar multiple of I and generates the IDENTICAL flow (Remark 6 of the
#     manuscript). The Jacobi rows are retained deliberately: matching residuals
#     with M = I are an empirical check of the scale-equivalence remark.
#   * Preconditioned (Ainv) runs use tight tolerances (abstol 1e-12, reltol
#     1e-10; they cost seconds); unpreconditioned runs keep standard tolerances
#     and are BUDGET-CENSORED (dt ~ 1/(α·λmax(A))): t_end records how far the
#     integration actually reached — that cost blow-up IS the conditioning story.
#   * PILOT-GATED: default dims include n=20 (to re-derive the Experiment-5
#     table under the honest reference residual); --large adds n=10000.
#   * τ_n = 1/λmax(A) analytic; cancellation-reduced R_ref + roundoff guard.
#   * Timing: setup (construction+factorization) and solve reported separately.
#   * --sensitivity: repeat with all tolerances ÷10 (5% stability check).
#
# Output: results/obstacle_finegrid/{raw*.csv, summary*.csv, manifest*.txt}
#         results/logs/s61_obstacle_finegrid_*.log
#         CANONICAL production artifact = the --large triple (raw_large.csv,
#         summary_large.csv, manifest_large.txt); all other suffixes are
#         pilots/diagnostics and must not be tabulated in the manuscript.
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s61_obstacle_finegrid.jl               # pilot dims
#   julia --project=. scripts/s61_obstacle_finegrid.jl --large       # + n=10000
#   julia --project=. scripts/s61_obstacle_finegrid.jl --topdims     # n=1000,10000 only
#                                    (overrides --quick/--large for DIMS; its output
#                                    supersedes overlapping dims from earlier files)
#   julia --project=. scripts/s61_obstacle_finegrid.jl --quick       # n=20,100 only
#   julia --project=. scripts/s61_obstacle_finegrid.jl --sensitivity # ÷10 audit
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics, SHA

# Run manifest, transactional: sources are hashed AT PROCESS START into a
# .pending file (a post-launch disk hash cannot prove which code a running
# process loaded), stale final CSVs for this suffix are deleted so a crash can
# never pair a fresh manifest with stale data, and finalize_manifest! appends
# the completion timestamp + CSV hashes and promotes .pending -> final ONLY on
# clean completion.
function write_manifest_start(results_dir, suffix, config::String)
    srcdir = joinpath(@__DIR__, "..", "src")
    files = vcat([@__FILE__],
                 [joinpath(srcdir, f) for f in
                  ["SPNNQVI.jl", "types.jl", "solver.jl", "projection.jl",
                   "problems.jl", "utils.jl", "io_utils.jl"]])
    for stale in ["raw$(suffix).csv", "summary$(suffix).csv", "manifest$(suffix).txt"]
        rm(joinpath(results_dir, stale); force=true)
    end
    open(joinpath(results_dir, "manifest$(suffix).pending.txt"), "w") do io
        println(io, "started: ", string(Base.Libc.strftime("%Y-%m-%dT%H:%M:%S", time())))
        println(io, "julia: ", string(VERSION))
        println(io, "args: ", join(ARGS, " "))
        println(io, "config: ", config)
        for f in files
            println(io, bytes2hex(sha256(read(f))), "  ", basename(f))
        end
    end
end

function finalize_manifest!(results_dir, suffix)
    pend = joinpath(results_dir, "manifest$(suffix).pending.txt")
    open(pend, "a") do io
        println(io, "completed: ", string(Base.Libc.strftime("%Y-%m-%dT%H:%M:%S", time())))
        for c in ["raw$(suffix).csv", "summary$(suffix).csv"]
            println(io, bytes2hex(sha256(read(joinpath(results_dir, c)))), "  ", c)
        end
    end
    mv(pend, joinpath(results_dir, "manifest$(suffix).txt"); force=true)
end

function main()
    LARGE       = "--large"       in ARGS
    QUICK       = "--quick"       in ARGS
    TOPDIMS     = "--topdims"     in ARGS      # n = 1000 and 10000 only (resume after cap fix)
    SENSITIVITY = "--sensitivity" in ARGS

    logpath, tee, logfile = setup_logging("s61_obstacle_finegrid")

    # ── Configuration ───────────────────────────────────────────────────────
    DIMS = TOPDIMS ? [1000, 10_000] :
           (QUICK ? [20, 100] : (LARGE ? [20, 100, 500, 1000, 10_000] : [20, 100, 500, 1000]))
    δ        = 0.1
    α_base   = 0.8
    λ        = 1.0
    T_FINAL  = 200.0
    EPS_ABS  = 1e-6                 # absolute RMS rule (reported; hard at large n)
    EPS_REL  = 1e-6                 # PRIMARY: relative RMS drop vs R_rms(x0)
    SAVE_DT  = 1.0
    MAXITER  = 200_000              # explicit-step budget (unpreconditioned runs censor here)
    WALL_CAP = 40.0                 # per-run wall cap for the CENSORED unpreconditioned runs
    KM_WALL_CAP = 900.0             # per-run wall cap for the converging A^-1 KM arm (its
                                    # iteration count grows with n as the contact set settles
                                    # mesh-dependently; n=1000 needs ~60 s, n=10^4 more)
    KM_H     = 0.1                  # KM relaxation step for the discrete A^-1 arm at n >= 500:
                                    # adaptive integration of the projected flow collapses at fine
                                    # grids (active-set kinks vs tight tolerances), so the A^-1 arm
                                    # there uses the discrete scheme of Prop. 2
                                    # (solve_qvi_fixedpoint — the KM discretization FORM of
                                    # Prop. 2, used as an integrator; the proposition's
                                    # contraction-window hypothesis does NOT hold for A^-1 at
                                    # fine grids, and success is certified by the R_ref stopping
                                    # rule, not by the proposition; free-space contraction factor
                                    # ~ 0.21 per iterate). NOTE: the KM arm is bounded by its iteration
                                    # budget (KM_H*KM_MAXIT model time), NOT by the common horizon
                                    # T_FINAL — model times are comparable only per solver (see the
                                    # solver column).
    KM_MAXIT = 200_000              # KM outer-iteration budget
    # Tolerances: at n = 20 (the tabulated comparison) ALL arms share the tight
    # pair, so the manuscript's identical-settings claim holds where it is
    # asserted; at fine grids the preconditioned arm keeps tight tolerances
    # (cheap) while the censored unpreconditioned arms use standard ones —
    # disclosed per arm in the CSV.
    ABS_STD  = SENSITIVITY ? 1e-9  : 1e-8
    REL_STD  = SENSITIVITY ? 1e-7  : 1e-6
    ABS_PRE  = SENSITIVITY ? 1e-13 : 1e-12
    REL_PRE  = SENSITIVITY ? 1e-11 : 1e-10
    METRICS  = [:identity, :jacobi, :Ainv]
    METRIC_LABELS = Dict(:identity => "M=I", :jacobi => "M=diag(A)^-1", :Ainv => "M=A^-1")
    N_STARTS = 4

    println(tee, "=" ^ 74)
    println(tee, "s61: Fine-grid obstacle QVI (sparse implementation)")
    println(tee, "  dims: ", DIMS, "   metrics: ", join(string.(METRICS), ", "))
    @printf(tee, "  delta=%.2f  alpha_base=%.2f  T=%.0f  rel eps=%.1e (primary)  abs eps=%.1e\n",
            δ, α_base, T_FINAL, EPS_REL, EPS_ABS)
    SENSITIVITY && println(tee, "  SENSITIVITY AUDIT: all tolerances divided by 10")
    println(tee, "=" ^ 74)

    results_dir = joinpath(@__DIR__, "..", "results", "obstacle_finegrid")
    mkpath(results_dir)
    suffix = (SENSITIVITY ? "_tighter" : "") * (QUICK ? "_quick" : "") * (LARGE ? "_large" : "") * (TOPDIMS ? "_topdims" : "")
    write_manifest_start(results_dir, suffix, @sprintf(
        "T_FINAL=%.0f KM_H=%.2f KM_MAXIT=%d MAXITER=%d WALL_CAP=%.0f KM_WALL_CAP=%.0f EPS_REL=%.0e EPS_ABS=%.0e tol_std=%.0e/%.0e tol_pre=%.0e/%.0e alpha_base=%.2f delta=%.2f lambda=%.1f",
        T_FINAL, KM_H, KM_MAXIT, MAXITER, WALL_CAP, KM_WALL_CAP,
        EPS_REL, EPS_ABS, ABS_STD, REL_STD, ABS_PRE, REL_PRE, α_base, δ, λ))

    raw_lines = String[]
    push!(raw_lines, "n,metric,start,solver,kappa_A,tau,alpha,abstol,reltol,t_setup,t_solve,t_end_model,t_hit_model,R_rms_initial,R_rms_final,rel_drop,hit_rel,hit_abs,dominant_rule,stop_reason,naccept,nreject,retcode,pdas_iters,pdas_cap_accepts,pdas_max_kkt,guard_unreliable,roundoff_ratio")
    summary_lines = String[]
    push!(summary_lines, "n,metric,solver,abstol,reltol,wall_cap,kappa_A,n_starts,success_rel,success_abs,t_setup,t_solve_median,t_hit_model_median,t_end_nonhit_median,R_rms_median,rel_drop_median")

    for n in DIMS
        h = 1.0 / (n + 1)
        λmax_A = (4.0 / h^2) * cos(π * h / 2)^2
        λmin_A = (4.0 / h^2) * sin(π * h / 2)^2
        κ_A = λmax_A / λmin_A
        τ = 1.0 / λmax_A
        grid = [i * h for i in 1:n]
        ψ0 = 0.2 * sin.(π * grid)

        # Starts (deterministic patterns, matching s60 including flat_0.5)
        starts = [0.5 * (ψ0 .+ 1.0), ψ0 .+ 0.01, ones(n), 0.5 * ones(n)][1:N_STARTS]
        start_labels = ["midway", "near_obstacle", "flat_1", "flat_0.5"][1:N_STARTS]

        @printf(tee, "\n--- n=%d  kappa(A)=%.3e  tau=%.3e ---\n", n, κ_A, τ)

        for met in METRICS
            # Setup timed separately (includes sparse Cholesky for :Ainv)
            t0 = time()
            prob0 = get_problem(5; n=n, δ=δ, metric=met, sparse_impl=true)
            t_setup = time() - t0
            α = α_base / norm_M(prob0)
            cfg = SolverConfig(T=T_FINAL, alpha=α, lambda=λ, tol=0.0, maxiter=MAXITER)
            precond = (met == :Ainv)
            tight = precond || n <= 20            # common tolerances at the tabulated n=20
            abstol = tight ? ABS_PRE : ABS_STD
            reltol = tight ? REL_PRE : REL_STD

            hit_rel_flags = Bool[]; hit_abs_flags = Bool[]
            t_solves = Float64[]; t_hits = Float64[]; t_ends = Float64[]
            r_rms_finals = Float64[]; rel_drops = Float64[]
            # At n >= 10^4, the unpreconditioned flows are pure known-outcome
            # censoring: run a single demonstration start; A^-1 runs all starts.
            met_start_idx = (n >= 10_000 && met != :Ainv) ? (1:1) : eachindex(starts)
            for sidx in met_start_idx
                x0 = starts[sidx]
                prob = with_x0(prob0, x0)
                R_rms0 = reference_residual_lb(x0, prob, τ, prob.lb) / sqrt(n)
                # Stop when EITHER rule fires (primary = relative; absolute reported),
                # or at the per-run wall cap (censored). dominant_rule records which
                # rule set the threshold; the WALL test precedes the residual test
                # (the cap defines censoring), and both stop causes are LATCHED so
                # a post-deadline observation is never credited as a hit.
                # NOTE: under the union stop, hit_abs can fire only when the run
                # overshoots the (looser) dominant threshold within one accepted
                # step — it reports overshoot, not absolute-rule attainability.
                rel_thresh = EPS_REL * R_rms0 * sqrt(n)
                abs_thresh = EPS_ABS * sqrt(n)
                thresh = max(rel_thresh, abs_thresh)
                dominant_rule = rel_thresh >= abs_thresh ? "rel" : "abs"
                dominant_rule == "abs" &&
                    println(tee, "  [note] absolute rule dominates for this start (R_rms0 = $(R_rms0))")

                # Roundoff guard (codex 002 §5.1), per start and against the
                # criterion actually used: u·‖x0 - m(x0)‖/τ vs the raw threshold
                guard = eps(1.0) * norm(x0 - prob0.m(x0)) / τ
                guard_ratio = guard / thresh
                if guard_ratio > 0.01
                    @printf(tee, "  [roundoff guard] u*||x0-m(x0)||/tau = %.2e = %.1f%% of the stop threshold %s\n",
                            guard, 100*guard_ratio,
                            guard_ratio > 0.5 ? "-- RESULT UNRELIABLE, needs higher precision" : "(acceptable)")
                end

                use_km = precond && n >= 500
                cap = use_km ? KM_WALL_CAP : WALL_CAP
                hit_latch = Ref(false); wall_latch = Ref(false)
                pdas_i0 = SPNNQVI.PDAS_ITER_COUNT[]
                pdas_c0 = SPNNQVI.PDAS_CAP_ACCEPTS[]
                SPNNQVI.PDAS_MAX_KKT[] = 0.0
                t1 = time()
                # The wall cap DEFINES censoring, so elapsed time is tested
                # FIRST: a below-threshold endpoint first observed after the
                # deadline is recorded as wall-censored, never as a hit.
                stop_check = x -> begin
                    if time() - t1 > cap
                        wall_latch[] = true; return true
                    end
                    if reference_residual_lb(x, prob, τ, prob.lb) <= thresh
                        hit_latch[] = true; return true
                    end
                    return false
                end
                local x_final, t_end, naccept, nreject, retcode, solver_tag
                if use_km
                    # Discrete KM scheme (Proposition 2 form) for the A^-1 arm
                    out = solve_qvi_fixedpoint(prob, cfg; h=KM_H,
                                               stop_fn=(x, k) -> stop_check(x), maxiter=KM_MAXIT)
                    x_final = out.x_final
                    t_end = KM_H * out.iterations
                    naccept = out.iterations; nreject = 0
                    # stopped=false with iterations < KM_MAXIT means the solver's
                    # divergence guard broke the loop, not the iteration budget
                    retcode = out.stopped ? "Stopped" :
                              (out.iterations >= KM_MAXIT ? "MaxIter" : "Diverged")
                    solver_tag = "km"
                else
                    ts, xs, rs, _, sol = solve_qvi_diffeq(prob, cfg; save_dt=SAVE_DT,
                        abstol=abstol, reltol=reltol, compute_rs=false,
                        terminate_on_tol=false, stop_fn=(u, t) -> stop_check(u), return_sol=true)
                    x_final = xs[end]
                    t_end = ts[end]
                    naccept = sol.stats.naccept
                    nreject = sol.stats.nreject
                    retcode = string(sol.retcode)
                    solver_tag = "tsit5"
                end
                t_solve = time() - t1

                R_raw = reference_residual_lb(x_final, prob, τ, prob.lb)
                R_rms = R_raw / sqrt(n)
                rel_drop = R_rms / R_rms0
                # Success is credited ONLY on a latched threshold hit; a run whose
                # final residual happens to pass but which stopped for another
                # reason is reported by its stop_reason, not as a hit. The hit
                # flags reuse the same raw thresholds as the stop callback, so a
                # latched hit always sets at least one flag.
                hit_rel = hit_latch[] && R_raw <= rel_thresh
                hit_abs = hit_latch[] && R_raw <= abs_thresh
                t_hit = hit_latch[] ? t_end : Inf
                stop_reason = hit_latch[]  ? "hit"    :
                              wall_latch[] ? "wall"   :
                              startswith(retcode, "MaxIter") ? "budget" :
                              retcode in ("Success", "Stopped", "Terminated") ? "horizon" :
                              "solver:" * retcode

                pdas_iters = SPNNQVI.PDAS_ITER_COUNT[] - pdas_i0
                pdas_caps = SPNNQVI.PDAS_CAP_ACCEPTS[] - pdas_c0
                pdas_maxk = SPNNQVI.PDAS_MAX_KKT[]
                pdas_caps > 0 && @printf(tee, "  [pdas] %d cap-path acceptance(s) this run -- investigate\n", pdas_caps)
                push!(hit_rel_flags, hit_rel); push!(hit_abs_flags, hit_abs)
                push!(t_solves, t_solve); push!(t_hits, t_hit); push!(t_ends, t_end)
                push!(r_rms_finals, R_rms); push!(rel_drops, rel_drop)
                push!(raw_lines, @sprintf("%d,%s,%s,%s,%.4e,%.4e,%.6e,%.1e,%.1e,%.3f,%.3f,%.3f,%.3f,%.4e,%.4e,%.4e,%d,%d,%s,%s,%d,%d,%s,%d,%d,%.3e,%d,%.3e",
                      n, met, start_labels[sidx], solver_tag, κ_A, τ, α, abstol, reltol,
                      t_setup, t_solve, t_end, t_hit,
                      R_rms0, R_rms, rel_drop, hit_rel, hit_abs, dominant_rule, stop_reason,
                      naccept, nreject, retcode, pdas_iters, pdas_caps, pdas_maxk,
                      guard_ratio > 0.5, guard_ratio))
                @printf(tee, "  %-14s %-13s  R_rms=%.2e  rel=%.1e  t_end=%7.2f  wall=%6.2fs  %s\n",
                        METRIC_LABELS[met], start_labels[sidx], R_rms, rel_drop, t_end, t_solve,
                        hit_rel ? (hit_abs ? "ok(rel+abs)" : "ok(rel)") :
                        hit_abs ? "ok(abs)" :
                        stop_reason == "wall"   ? "WALL-CENSORED" :
                        stop_reason == "budget" ? "BUDGET-CENSORED at t_end" : "NOT CONVERGED")
            end

            hits_t = [t for (t, f) in zip(t_solves, hit_rel_flags) if f]
            hits_m = [t for (t, f) in zip(t_hits, hit_rel_flags) if f]
            nonhit_e = [t for (t, f) in zip(t_ends, hit_rel_flags) if !f]
            smet = (met == :Ainv && n >= 500) ? "km" : "tsit5"
            scap = (met == :Ainv && n >= 500) ? KM_WALL_CAP : WALL_CAP
            stight = (met == :Ainv || n <= 20)
            push!(summary_lines, @sprintf("%d,%s,%s,%.1e,%.1e,%.0f,%.4e,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4e,%.4e",
                  n, met, smet,
                  stight ? ABS_PRE : ABS_STD, stight ? REL_PRE : REL_STD, scap,
                  κ_A, length(met_start_idx),
                  mean(hit_rel_flags), mean(hit_abs_flags), t_setup,
                  isempty(hits_t) ? NaN : median(hits_t),
                  isempty(hits_m) ? NaN : median(hits_m),
                  isempty(nonhit_e) ? NaN : median(nonhit_e),
                  median(r_rms_finals), median(rel_drops)))
        end
    end

    open(joinpath(results_dir, "raw$(suffix).csv"), "w") do io
        foreach(l -> println(io, l), raw_lines)
    end
    open(joinpath(results_dir, "summary$(suffix).csv"), "w") do io
        foreach(l -> println(io, l), summary_lines)
    end
    finalize_manifest!(results_dir, suffix)

    println(tee, "\nResults saved to: $results_dir")
    println(tee, "Largest VERIFIED dimension = largest n with success_rel = 1.0 for M = A^-1.")
    println(tee, "Report that dimension in the paper — do not promise beyond it.")
    println(tee, "NOTE: M=I and Jacobi rows are expected to MATCH exactly (constant diagonal")
    println(tee, "=> Jacobi is a scalar rescaling; see Remark 6) and to be budget-censored at")
    println(tee, "large n (dt ~ 1/(alpha*lambda_max(A))): that cost blow-up is the finding.")
    teardown_logging(tee, logpath)
end

main()
