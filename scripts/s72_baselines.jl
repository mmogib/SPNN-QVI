# ============================================================================
# s72: Baseline comparison — SPNN vs Noor PDS vs discrete projection iterations
# ============================================================================
#
# Goal:   Compare four methods on three problems under a COMMON method-independent
#         reference residual (R3.3 of the Revision 1 response):
#           1. spnn_scaled  — SPNN dynamics (Tsit5), predetermined recipe metric
#           2. noor_pds     — SPNN dynamics (Tsit5), M = I  (≡ Noor's implicit PDS)
#           3. picard_mI    — fixed-point iteration x_{k+1}=T(x_k), M = I
#                             (= Noor 1988, Algorithm 3.1 with g = id; Prop. 2, h=1/λ)
#           4. picard_scaled— fixed-point iteration, recipe metric
#         Problems: P5 obstacle (n=20), P7 nonlinear (n=5), P8 scaling (n=50).
#
# Protocol (agreed with the co-reviewer, channels/codex_to_claude/002 §5):
#   * Success/stopping for EVERY method: R_ref(x) ≤ EPS_REF, evaluated at accepted
#     ODE endpoints / completed outer iterates; all native convergence rules OFF.
#   * τ analytic per problem (τ = 1/K, never tuned): P5: 1/λmax(A) (closed form);
#     P7: 1/11; P8: 1/opnorm(Q_op) computed once outside timed regions.
#   * α tuning: equal log grid per method; tuned on PILOT starts (lock by success
#     rate, then median end-to-end time), evaluated on held-out starts.
#   * Counts: complete map evaluations — each evaluation of T performs exactly
#     one F evaluation and one metric projection by construction, so the F
#     count IS the complete projection count; PDAS inner iterations (the A⁻¹
#     arm's inner solves) are tallied separately, as are reference checks.
#   * Budgets: DUAL formulation-specific budgets (ODE arms: model-time horizon
#     + step budget; discrete arms: outer-iteration cap) plus a COMMON SOFT
#     wall-clock check, polled at accepted endpoints / completed iterates and
#     therefore not hard-enforced. Success rates are NOT comparable as
#     outcomes under one binding budget; stop reasons and timeout counts are
#     reported so each cell's terminating budget is identifiable.
#   * Timing: per-run end-to-end solve time includes all reference checks
#     (initial and final included) and EXCLUDES the separately reported setup
#     time; Julia warm-up before timing; method order randomized per start;
#     median/IQR over held-out starts; timeouts censored.
#   * --tighter reuses the production locks (never retunes) and runs an
#     automated paired gate: exact per-start classification/termination tuples
#     are a hard requirement; median-time changes beyond 5% are reported as a
#     sensitivity diagnostic because tighter ODE tolerances legitimately change
#     accepted-step and map-evaluation counts.
#
# Output: results/baselines/raw.csv, results/baselines/summary.csv
#         results/logs/s72_baselines_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s72_baselines.jl            # full run
#   julia --project=. scripts/s72_baselines.jl --quick    # reduced grid/starts
#   julia --project=. scripts/s72_baselines.jl --tighter  # abstol/reltol ÷10 audit
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Random, Statistics, SHA

# Run manifest, transactional: sources hashed at process start into .pending;
# stale final CSVs for this suffix deleted so a crash never pairs a fresh
# manifest with stale data; finalize appends completion time + CSV hashes and
# promotes .pending -> final only on clean completion.
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
    QUICK   = "--quick"   in ARGS
    TIGHTER = "--tighter" in ARGS
    QUICK && TIGHTER && error("--quick and --tighter are mutually exclusive " *
                              "(the audit must pair with the production run)")

    logpath, tee, logfile = setup_logging("s72_baselines")

    # ── Configuration ───────────────────────────────────────────────────────
    EPS_REF      = 1e-6                   # raw R_ref success threshold (fixed n per problem)
    LAMBDA       = 1.0
    T_BUDGET     = 500.0                  # ODE model-time budget
    MAXIT_FP     = 200_000                # fixed-point iteration budget
    WALL_LIMIT   = 60.0                   # per-run wall timeout (seconds)
    ABSTOL       = TIGHTER ? 1e-9 : 1e-8
    RELTOL       = TIGHTER ? 1e-7 : 1e-6
    # The first production gate locked two P5 methods at 1e-3. Extend the
    # lower end by one decade at the same quarter-decade spacing; all other
    # tuning rules and the upper endpoint remain unchanged.
    ALPHA_GRID   = QUICK ? (10.0 .^ range(-3, 1, length=9)) : (10.0 .^ range(-4, 1, length=21))
    N_PILOT      = 3
    N_HELDOUT    = QUICK ? 3 : 7
    SEED_STARTS  = 20260722

    # Problems and predetermined recipe metrics (Jacobi/diagonal scaling of the
    # symmetric part — the deterministic recipe of the revised Section 4 remark).
    # τ = 1/K with K a global Euclidean Lipschitz constant of the UNSCALED F.
    h20 = 1.0 / 21
    K_P5 = (4.0 / h20^2) * cos(π * h20 / 2)^2          # λmax of A, closed form
    PROBLEMS = [
        # NOTE: for the 1-D obstacle problem diag(A) is CONSTANT, so :jacobi is a
        # scalar rescaling of I (identical flow, Remark 6). The genuine
        # preconditioner for this problem is the operator inverse :Ainv.
        (key="P5_obstacle",  id=5, kwargs=(n=20, sparse_impl=true),  recipe=:Ainv,  K=K_P5,
         start_fn = rng -> begin
             g = [i*h20 for i in 1:20]; ψ0 = 0.2*sin.(π*g)
             ψ0 .+ 0.05 .+ 0.95 .* rand(rng, 20)
         end),
        (key="P7_nonlinear", id=7, kwargs=(n=5,),   recipe=:diag_inv, K=11.0,
         start_fn = rng -> 5.0 .* rand(rng, 5)),
        (key="P8_scaling",   id=8, kwargs=(n=50,),  recipe=:diag_inv, K=NaN,   # filled below
         start_fn = rng -> 10.0 .* rand(rng, 50)),
    ]

    METHODS = ["spnn_scaled", "noor_pds", "picard_mI", "picard_scaled"]

    println(tee, "=" ^ 74)
    println(tee, "s72: Baseline comparison — common reference residual R_ref ≤ $(EPS_REF)")
    println(tee, "  methods: ", join(METHODS, ", "))
    @printf(tee, "  alpha grid: %d points in [%.0e, %.0e]\n",
            length(ALPHA_GRID), ALPHA_GRID[1], ALPHA_GRID[end])
    println(tee, "  pilot starts: $(N_PILOT), held-out starts: $(N_HELDOUT)")
    TIGHTER && println(tee, "  SENSITIVITY AUDIT: abstol=$(ABSTOL), reltol=$(RELTOL)")
    println(tee, "=" ^ 74)

    results_dir = joinpath(@__DIR__, "..", "results", "baselines")
    mkpath(results_dir)
    SUFFIX = TIGHTER ? "_tighter" : (QUICK ? "_quick" : "")
    write_manifest_start(results_dir, SUFFIX, @sprintf(
        "EPS_REF=%.0e LAMBDA=%.1f T_BUDGET=%.0f MAXIT_FP=%d WALL_LIMIT=%.0f ABSTOL=%.0e RELTOL=%.0e grid=%d@[%.0e,%.0e] N_PILOT=%d N_HELDOUT=%d SEED=%d",
        EPS_REF, LAMBDA, T_BUDGET, MAXIT_FP, WALL_LIMIT, ABSTOL, RELTOL,
        length(ALPHA_GRID), ALPHA_GRID[1], ALPHA_GRID[end], N_PILOT, N_HELDOUT, SEED_STARTS))
    boundary_locks = String[]
    gate_failed = false

    # ── Helpers ─────────────────────────────────────────────────────────────

    # Wrap a problem with an F-evaluation counter. Each evaluation of the map T
    # performs exactly one F evaluation and one metric projection (see T_map),
    # so the F count equals the complete map/projection-evaluation count; the
    # A⁻¹ arm's PDAS inner iterations are tallied via SPNNQVI.PDAS_ITER_COUNT.
    function counted_problem(prob)
        nF = Ref(0)
        F_c(x) = (nF[] += 1; prob.F(x))
        p = QVIProblem(F=F_c, m=prob.m, proj_S=prob.proj_S, M=prob.M, x0=prob.x0,
                       n=prob.n, name=prob.name, Minv=prob.Minv,
                       lb=prob.lb, ub=prob.ub, Mnorm=prob.Mnorm)
        return p, nF
    end

    # One run of one method at one alpha_base from one start.
    # Returns (hit, t, ode_naccept, ode_nreject, fp_iters, map_evals,
    #          pdas_iters, R_checks, R_final, timeout, retcode)
    function run_one(method, prob0, x0, alpha_base, tau)
        prob = with_x0(prob0, x0)
        Mn = norm_M(prob)
        alpha = alpha_base / Mn
        cfg = SolverConfig(T=T_BUDGET, alpha=alpha, lambda=LAMBDA,
                           tol=0.0, maxiter=MAXIT_FP)     # tol=0: native rule off
        cprob, nF = counted_problem(prob)
        pdas0 = SPNNQVI.PDAS_ITER_COUNT[]

        hit = Ref(false); timeout = Ref(false); nRef = Ref(0)
        t0 = time()
        # Reference checks use the UNCOUNTED problem; the wall limit is tested
        # FIRST so a post-limit hit is never credited; R_checks counts only
        # ACTUAL residual evaluations (a timeout-fired callback skips the
        # evaluation and is not counted).
        check = x -> begin
            if time() - t0 > WALL_LIMIT
                timeout[] = true; return true
            end
            nRef[] += 1
            if reference_residual(x, prob, tau) <= EPS_REF
                hit[] = true; return true
            end
            return false
        end

        # Immediate check at the initial point (counted; elapsed time returned
        # so at-start hits still include the cost of that reference check)
        R0 = reference_residual(x0, prob, tau)
        nRef[] += 1
        if R0 <= EPS_REF
            return (hit=true, t=time() - t0, ode_naccept=0, ode_nreject=0, fp_iters=0,
                    map_evals=0, pdas_iters=0, R_checks=nRef[], R_final=R0,
                    timeout=false, retcode="AtStart")
        end

        local x_final, ode_naccept, ode_nreject, fp_iters, retcode
        if method in ("spnn_scaled", "noor_pds")
            ts, xs, rs, _, sol = solve_qvi_diffeq(cprob, cfg; save_dt=1.0,
                abstol=ABSTOL, reltol=RELTOL, compute_rs=false,
                terminate_on_tol=false, stop_fn=(u, t) -> check(u), return_sol=true)
            x_final = xs[end]
            ode_naccept = Int(sol.stats.naccept); ode_nreject = Int(sol.stats.nreject)
            fp_iters = 0; retcode = string(sol.retcode)
        else
            out = solve_qvi_fixedpoint(cprob, cfg; h=1.0/LAMBDA,
                                       stop_fn=(x, k) -> check(x), maxiter=MAXIT_FP)
            x_final = out.x_final; fp_iters = out.iterations
            ode_naccept = 0; ode_nreject = 0
            retcode = out.stopped ? "Stopped" :
                      (out.iterations >= MAXIT_FP ? "MaxIter" : "Diverged")
        end
        # The final residual check is part of the reported end-to-end time
        Rf = reference_residual(x_final, prob, tau)
        nRef[] += 1
        t_end = time() - t0
        return (hit=hit[] && !timeout[], t=t_end,
                ode_naccept=ode_naccept, ode_nreject=ode_nreject, fp_iters=fp_iters,
                map_evals=nF[], pdas_iters=SPNNQVI.PDAS_ITER_COUNT[] - pdas0,
                R_checks=nRef[], R_final=Rf, timeout=timeout[], retcode=retcode)
    end

    # Metric for a method on a problem spec
    metric_for(spec, method) = method in ("noor_pds", "picard_mI") ? :identity : spec.recipe

    # ── Main loop ───────────────────────────────────────────────────────────
    raw_lines = String[]
    push!(raw_lines, "problem,method,phase,start_id,alpha_base,hit,timeout,R_final,ode_naccept,ode_nreject,fp_iters,map_evals,pdas_iters,R_checks,t_seconds,retcode")
    summary_lines = String[]
    push!(summary_lines, "problem,method,alpha_locked,tau,success_rate,t_median,t_iqr_lo,t_iqr_hi,map_evals_median,ode_naccept_median,fp_iters_median,R_final_median_all,timeouts,n_heldout,stop_mix,t_setup")

    for spec in PROBLEMS
        # Setup (outside all timed regions): one problem instance per UNIQUE
        # metric, shared by the methods that use it; K, tau, starts
        t_setup0 = time()
        mprobs = Dict(mq => get_problem(spec.id; spec.kwargs..., metric=mq)
                      for mq in unique([metric_for(spec, met) for met in METHODS]))
        probs = Dict(met => mprobs[metric_for(spec, met)] for met in METHODS)
        K = spec.K
        if isnan(K)   # P8: K = opnorm(Q_op), computed once from the F Jacobian action
            p_id = probs["noor_pds"]
            n = p_id.n
            Qcols = [p_id.F(Float64.(1:n .== j)) - p_id.F(zeros(n)) for j in 1:n]
            K = opnorm(hcat(Qcols...))
        end
        tau = 1.0 / K
        t_setup = time() - t_setup0

        rng = MersenneTwister(SEED_STARTS)
        pilot_starts   = [spec.start_fn(rng) for _ in 1:N_PILOT]
        heldout_starts = [spec.start_fn(rng) for _ in 1:N_HELDOUT]

        @printf(tee, "\n--- %-12s  K=%.4e  tau=%.4e  setup=%.2fs ---\n",
                spec.key, K, tau, t_setup)

        # Warm-up (compile paths; discarded)
        for met in METHODS
            run_one(met, probs[met], pilot_starts[1], ALPHA_GRID[end÷2], tau)
        end

        locked = Dict{String,Float64}()
        if TIGHTER
            # Paired sensitivity audit: reuse the production locks, do NOT retune
            locked_file = joinpath(results_dir, "summary.csv")
            isfile(locked_file) || error("--tighter requires a prior standard run (summary.csv missing)")
            for line in readlines(locked_file)[2:end]
                f = split(line, ",")
                (length(f) >= 3 && f[1] == spec.key) || continue
                locked[String(f[2])] = parse(Float64, f[3])
            end
            for met in METHODS
                haskey(locked, met) || error("--tighter: no production lock for " *
                    "($(spec.key), $met) in summary.csv — rerun the standard run first")
            end
            println(tee, "  [--tighter] reusing locked alpha values from the standard summary.csv")
        else
        for met in METHODS
            # ── Pilot: tune alpha_base on pilot starts ──
            best = (rate=-1.0, tmed=Inf, a=ALPHA_GRID[1])
            for a in ALPHA_GRID
                outs = [run_one(met, probs[met], x0, a, tau) for x0 in pilot_starts]
                for (sid, o) in enumerate(outs)
                    push!(raw_lines, @sprintf("%s,%s,pilot,%d,%.6e,%d,%d,%.4e,%d,%d,%d,%d,%d,%d,%.4f,%s",
                        spec.key, met, sid, a, o.hit, o.timeout, o.R_final,
                        o.ode_naccept, o.ode_nreject, o.fp_iters,
                        o.map_evals, o.pdas_iters, o.R_checks, o.t, o.retcode))
                end
                rate = mean(o.hit for o in outs)
                hits_t = [o.t for o in outs if o.hit]
                tmed = isempty(hits_t) ? Inf : median(hits_t)
                if rate > best.rate || (rate == best.rate && tmed < best.tmed)
                    best = (rate=rate, tmed=tmed, a=a)
                end
            end
            locked[met] = best.a
            at_boundary = best.a == ALPHA_GRID[1] || best.a == ALPHA_GRID[end]
            at_boundary && push!(boundary_locks, "$(spec.key)/$met at $(best.a)")
            @printf(tee, "  %-14s locked alpha_base=%.4e (pilot success %.0f%%)%s\n",
                    met, best.a, 100*best.rate,
                    at_boundary ? "  [GATE FAIL: locked at grid boundary -- expand grid]" : "")

            # ── Held-out evaluation, randomized order handled below ──
        end
        end  # if TIGHTER

        # Held-out: randomize method order per start
        rng_order = MersenneTwister(SEED_STARTS + 1)
        heldout = Dict(met => NamedTuple[] for met in METHODS)
        for (sid, x0) in enumerate(heldout_starts)
            for met in shuffle(rng_order, METHODS)
                o = run_one(met, probs[met], x0, locked[met], tau)
                push!(heldout[met], o)
                push!(raw_lines, @sprintf("%s,%s,heldout,%d,%.6e,%d,%d,%.4e,%d,%d,%d,%d,%d,%d,%.4f,%s",
                    spec.key, met, sid, locked[met], o.hit, o.timeout, o.R_final,
                    o.ode_naccept, o.ode_nreject, o.fp_iters,
                    o.map_evals, o.pdas_iters, o.R_checks, o.t, o.retcode))
            end
        end

        for met in METHODS
            outs = heldout[met]
            hits = [o for o in outs if o.hit]
            rate = mean(o.hit for o in outs)
            ts_h = sort([o.t for o in hits])
            tmed = isempty(ts_h) ? NaN : median(ts_h)
            tlo  = isempty(ts_h) ? NaN : quantile(ts_h, 0.25)
            thi  = isempty(ts_h) ? NaN : quantile(ts_h, 0.75)
            fmed = isempty(hits) ? NaN : median([o.map_evals for o in hits])
            amed = isempty(hits) ? NaN : median([o.ode_naccept for o in hits])
            pmed = isempty(hits) ? NaN : median([o.fp_iters for o in hits])
            rfall = median([o.R_final for o in outs])       # over ALL held-out starts
            ntim = count(o.timeout for o in outs)
            # per-cell termination-cause mix (hit / timeout / budget / diverged / other)
            stop_of(o) = o.hit ? "hit" : o.timeout ? "timeout" :
                         startswith(o.retcode, "MaxIter") ? "budget" :
                         o.retcode == "Diverged" ? "diverged" : "other"
            mixcounts = Dict{String,Int}()
            for o in outs
                mixcounts[stop_of(o)] = get(mixcounts, stop_of(o), 0) + 1
            end
            stop_mix = join(["$k:$v" for (k, v) in sort(collect(mixcounts))], ";")
            # alpha_locked written full-precision so --tighter reuses the exact Float64
            push!(summary_lines, @sprintf("%s,%s,%.17e,%.6e,%.3f,%.4f,%.4f,%.4f,%.1f,%.1f,%.1f,%.4e,%d,%d,%s,%.3f",
                  spec.key, met, locked[met], tau, rate, tmed, tlo, thi, fmed, amed, pmed,
                  rfall, ntim, length(outs), stop_mix, t_setup))
            @printf(tee, "  %-14s success %5.1f%%  t_med=%7.3fs  map_evals_med=%9.0f  timeouts=%d\n",
                    met, 100*rate, tmed, fmed, ntim)
        end
    end

    # ── Save (quick output is suffixed so it can never poison the production
    #     locks that --tighter reads from summary.csv) ─────────────────────────
    open(joinpath(results_dir, "raw$(SUFFIX).csv"), "w") do io
        foreach(l -> println(io, l), raw_lines)
    end
    open(joinpath(results_dir, "summary$(SUFFIX).csv"), "w") do io
        foreach(l -> println(io, l), summary_lines)
    end

    # ── Automated paired sensitivity gate (--tighter): (1) exact key-set
    #    equality against the expected design and the production run (silent
    #    skips of missing keys could otherwise false-pass an empty overlap);
    #    (2) per-start (hit, timeout, retcode) tuples; (3) medians only if the
    #    classifications are unchanged. Key/classification failures raise AFTER
    #    files are written; timing changes are retained as diagnostic warnings
    #    rather than being forced to pass by repeated wall-time sampling. ─────
    if TIGHTER
        println(tee, "\n[--tighter gate] paired per-start comparison against production raw.csv:")
        expected = Set((spec.key, met, string(sid))
                       for spec in PROBLEMS, met in METHODS, sid in 1:N_HELDOUT)
        parse_heldout(lines) = begin
            d = Dict{Tuple{String,String,String},Tuple{String,String,String}}()
            dup = false
            for line in lines
                f = split(line, ",")
                (length(f) >= 16 && f[3] == "heldout") || continue
                key = (String(f[1]), String(f[2]), String(f[4]))
                haskey(d, key) && (dup = true)
                d[key] = (String(f[6]), String(f[7]), String(f[16]))
            end
            d, dup
        end
        prod, prod_dup = parse_heldout(readlines(joinpath(results_dir, "raw.csv"))[2:end])
        cur, cur_dup = parse_heldout(raw_lines[2:end])
        if prod_dup || cur_dup || Set(keys(prod)) != expected || Set(keys(cur)) != expected
            gate_failed = true
            println(tee, "[--tighter gate] FAIL: held-out key sets do not exactly match the " *
                "expected design ($(length(expected)) keys; production has $(length(prod)), " *
                "this run has $(length(cur)); duplicates prod=$prod_dup cur=$cur_dup).")
        else
            nchanged = 0
            for key in sort(collect(expected))
                if prod[key] != cur[key]
                    nchanged += 1
                    println(tee, "  MISMATCH: $(key[1])/$(key[2]) start $(key[3]): " *
                        "(hit,timeout,retcode) $(prod[key]) -> $(cur[key])")
                end
            end
            @printf(tee, "  per-start (hit, timeout, retcode) tuples: %d matched, %d changed\n",
                    length(expected) - nchanged, nchanged)
            if nchanged > 0
                gate_failed = true
                println(tee, "[--tighter gate] FAIL: per-start classifications changed -- medians not compared.")
            else
                prodsum = Dict{Tuple{String,String},Vector{String}}()
                for line in readlines(joinpath(results_dir, "summary.csv"))[2:end]
                    f = split(line, ",")
                    prodsum[(String(f[1]), String(f[2]))] = String.(f)
                end
                ntiming = 0; ncomp = 0
                for line in summary_lines[2:end]
                    f = split(line, ",")
                    key = (String(f[1]), String(f[2]))
                    haskey(prodsum, key) || (gate_failed = true; continue)
                    ncomp += 1
                    p = prodsum[key]
                    tmed_t = parse(Float64, f[6]); tmed_p = parse(Float64, p[6])
                    dt_ok = (isnan(tmed_t) && isnan(tmed_p)) ||
                            (!isnan(tmed_t) && !isnan(tmed_p) &&
                             abs(tmed_t - tmed_p) <= 0.05 * tmed_p)
                    dt_ok || (ntiming += 1)
                    @printf(tee, "  %-12s %-14s t_med %.4f -> %.4f  %s\n",
                            key[1], key[2], tmed_p, tmed_t,
                            dt_ok ? "PASS" : "TIMING-WARN")
                end
                ncomp == length(PROBLEMS) * length(METHODS) || (gate_failed = true)
                if gate_failed
                    println(tee, "[--tighter gate] FAIL: summary cells missing.")
                else
                    println(tee, "[--tighter gate] PASS: all per-start classifications " *
                        "and termination tuples unchanged across $(ncomp) cells.")
                    println(tee, ntiming == 0 ?
                        "[--tighter timing diagnostic] all $(ncomp) median times within 5%." :
                        "[--tighter timing diagnostic] WARN: $ntiming median-time cell(s) " *
                        "beyond 5%; production timings remain canonical.")
                end
            end
        end
    end

    gate_error = if !isempty(boundary_locks)
        "PRODUCTION GATE FAIL: alpha locked at a grid boundary for: " *
        join(boundary_locks, "; ") * " -- expand ALPHA_GRID and rerun."
    elseif gate_failed
        "SENSITIVITY GATE FAIL: --tighter comparison did not pass (see log); " *
        "outputs retained for diagnosis."
    else
        nothing
    end

    println(tee, "\nResults saved to: $results_dir")
    println(tee, "  raw$(SUFFIX).csv (per-run), summary$(SUFFIX).csv (per problem × method)")
    println(tee, "NOTE: timeouts are censored failures — never averaged into times.")
    println(tee, "NOTE: per-run solve time includes all reference checks (initial and")
    println(tee, "final included) and EXCLUDES the separately reported setup time.")
    if !isnothing(gate_error)
        println(tee, "ERROR: $gate_error")
        println(tee, "NOTE: diagnostic CSVs and the pending manifest are retained; " *
                     "no FINAL manifest was promoted.")
    end
    teardown_logging(tee, logpath)

    # Gate failures raise AFTER diagnostic outputs are written and logged.
    # Only a gate-clean, normally closed run may promote the pending manifest.
    isnothing(gate_error) || error(gate_error)
    finalize_manifest!(results_dir, SUFFIX)
end

main()
