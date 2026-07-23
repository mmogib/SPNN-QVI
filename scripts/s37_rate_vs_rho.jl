# ============================================================================
# s37: Observed decay rate vs the certified rate λ(1-ρ_M)  (R3.2 / B5)
# ============================================================================
#
# Goal:   Quantify the predictive value of the ρ_M certificate on a family where
#         its window is NONEMPTY: Problem 1 with small κ and δ, M = I, so that
#           LK_m = δ < 1  and  μ_eff = 1 - δκ > 2κ√δ.
#         For each (κ, δ, α): check the envelope ‖e(t)‖ ≤ ‖e(0)‖e^{-λ(1-ρ_M)t}
#         (the certified bound; primary object) and fit the observed decay rate
#         by predeclared OLS on log‖e(t)‖ (secondary summary).
#
# Protocol (channels/codex_to_claude/002 §2.2):
#   * Exact solution x̄=(0.5,0.3) by construction (F(x̄)=0, interior).
#   * Metric representative fixed in advance: M = I (L = 1). Analytic μ=1, K=κ,
#     K_m=δ — never fitted.
#   * All early stopping disabled; predeclared uniform grid via saveat;
#     unweighted OLS on log‖e(t_j)‖ over the predeclared window
#     ‖e‖ ∈ [FLOOR, 0.1·‖e(0)‖]; ≥ MIN_SAMPLES or "unavailable".
#   * Fitting the NORM (not V_M): compare ĉ against λ(1-ρ_M) directly.
#   * For ρ_M ≥ 1 the certificate is silent (no growth prediction).
#
# This is a RESPONSE-LETTER figure only ("Response Figure R3.2"); the CSV and
# script are archived in the reproducibility package, not the manuscript.
#
# Output: results/rate_vs_rho/rates.csv  (+ fig_rate_vs_rho.pdf with --figure)
#         results/logs/s37_rate_vs_rho_*.log
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s37_rate_vs_rho.jl
#   julia --project=. scripts/s37_rate_vs_rho.jl --figure
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using LinearAlgebra, Printf, Statistics

# Plots is loaded only when the figure is requested (top-level conditional:
# `using` is not allowed inside functions).
if "--figure" in ARGS
    using Plots
end

function main()
    FIGURE  = "--figure" in ARGS
    TIGHTER = "--tighter" in ARGS      # paired audit: tolerances ÷10, same protocol

    logpath, tee, logfile = setup_logging("s37_rate_vs_rho")
    ABSTOL = TIGHTER ? 1e-9 : 1e-8
    RELTOL = TIGHTER ? 1e-7 : 1e-6

    # ── Predeclared protocol constants ──────────────────────────────────────
    KAPPAS      = [1.2, 1.5, 2.0]
    DELTAS      = [0.02, 0.05]
    λ           = 1.0
    T_FINAL     = 40.0
    SAVE_DT     = 0.01               # uniform sampling grid (dense output)
    FLOOR       = 1e-7               # fitting floor on ‖e‖
    HEAD_FRAC   = 0.1                # fit starts once ‖e‖ ≤ HEAD_FRAC·‖e(0)‖
    MIN_SAMPLES = 20
    XBAR        = [0.5, 0.3]
    N_INSIDE    = 6                  # α values inside the window
    ENV_SLACK   = 1e-8               # numerical slack for the envelope check

    println(tee, "=" ^ 74)
    println(tee, "s37: observed decay rate vs certified rate λ(1-ρ_M)   [Response Fig. R3.2]")
    println(tee, "  family: Problem 1, M = I;  κ ∈ $(KAPPAS), δ ∈ $(DELTAS)")
    println(tee, "=" ^ 74)

    results_dir = joinpath(@__DIR__, "..", "results", "rate_vs_rho")
    mkpath(results_dir)

    # ρ_M for M = I (L = 1): ρ_M = δ + sqrt((1+δ)² - 2αμ_eff + α²κ²), μ_eff = 1 - δκ
    rho_M(α, κ, δ) = δ + sqrt(max((1 + δ)^2 - 2α * (1 - δ*κ) + α^2 * κ^2, 0.0))

    lines = String[]
    push!(lines, "kappa,delta,alpha,in_window,rho_M,rate_pred,rate_fit,n_fit,envelope_ok")

    for κ in KAPPAS, δ in DELTAS
        μ_eff = 1 - δ * κ
        window_ok = (δ < 1) && (μ_eff > 2 * κ * sqrt(δ))
        if !window_ok
            @printf(tee, "\n(κ=%.2f, δ=%.2f): window EMPTY — certificate silent; skipping fits\n", κ, δ)
            continue
        end
        disc = sqrt(μ_eff^2 - 4 * κ^2 * δ)
        α_lo = (μ_eff - disc) / κ^2
        α_hi = (μ_eff + disc) / κ^2
        @printf(tee, "\n(κ=%.2f, δ=%.2f): μ_eff=%.4f  window=(%.4f, %.4f)\n", κ, δ, μ_eff, α_lo, α_hi)

        # α grid: N_INSIDE log-spaced inside [1.05α_lo, 0.95α_hi], plus two outside
        αs_in = exp.(range(log(1.05 * α_lo), log(0.95 * α_hi), length=N_INSIDE))
        αs = vcat(αs_in, [0.5 * α_lo, 1.5 * α_hi])

        for α in αs
            in_win = α_lo < α < α_hi
            ρ = rho_M(α, κ, δ)
            rate_pred = in_win ? λ * (1 - ρ) : NaN     # certificate silent outside

            prob = get_problem(1; κ=κ, δ=δ, metric=:identity)
            cfg = SolverConfig(T=T_FINAL, alpha=α, lambda=λ, tol=0.0)
            ts, xs, _, _ = solve_qvi_diffeq(prob, cfg; save_dt=SAVE_DT,
                                            abstol=ABSTOL, reltol=RELTOL,
                                            terminate_on_tol=false,
                                            compute_rs=false, dtmax_override=0.1)
            es = [norm(x - XBAR) for x in xs]
            e0 = es[1]

            # Envelope check (primary): ‖e(t)‖ ≤ e0·exp(-λ(1-ρ)t) + slack, when ρ<1
            envelope_ok = true
            if in_win && ρ < 1
                for (t, e) in zip(ts, es)
                    if e > e0 * exp(-λ * (1 - ρ) * t) + ENV_SLACK
                        envelope_ok = false
                        break
                    end
                end
            end

            # OLS fit (secondary) on the predeclared window
            idx = [i for i in eachindex(es) if FLOOR <= es[i] <= HEAD_FRAC * e0]
            local rate_fit, n_fit
            n_fit = length(idx)
            if n_fit >= MIN_SAMPLES
                tsf = ts[idx]; ysf = log.(es[idx])
                tm = mean(tsf); ym = mean(ysf)
                slope = sum((tsf .- tm) .* (ysf .- ym)) / sum((tsf .- tm) .^ 2)
                rate_fit = -slope
            else
                rate_fit = NaN      # "unavailable" — never adjust the window
            end

            push!(lines, @sprintf("%.3f,%.3f,%.6e,%d,%.6f,%.6f,%.6f,%d,%s",
                  κ, δ, α, in_win, ρ, rate_pred, rate_fit, n_fit,
                  (in_win && ρ < 1) ? string(Int(envelope_ok)) : "NA"))
            @printf(tee, "  α=%.4f %s ρ_M=%.4f  pred=%.4f  fit=%s (n=%d)  env=%s\n",
                    α, in_win ? "(in) " : "(out)", ρ,
                    in_win ? rate_pred : NaN,
                    isnan(rate_fit) ? "unavailable" : @sprintf("%.4f", rate_fit),
                    n_fit, (in_win && ρ < 1) ? (envelope_ok ? "ok" : "VIOLATED") : "NA")
        end
    end

    csvname = TIGHTER ? "rates_tighter.csv" : "rates.csv"
    open(joinpath(results_dir, csvname), "w") do io
        foreach(l -> println(io, l), lines)
    end
    println(tee, "\nCSV saved to: $(joinpath(results_dir, csvname))")
    TIGHTER && println(tee, "Paired audit: compare rate_fit vs rates.csv (5% rule).")

    # ── Optional response figure ────────────────────────────────────────────
    if FIGURE
        rows = [split(l, ",") for l in lines[2:end]]
        pred = [parse(Float64, r[6]) for r in rows]
        fit  = [parse(Float64, r[7]) for r in rows]
        keep = .!isnan.(pred) .& .!isnan.(fit)
        plt = scatter(pred[keep], fit[keep];
                      xlabel="certified rate  λ(1-ρ_M)", ylabel="fitted decay rate",
                      label="runs (window nonempty)", legend=:topleft, dpi=200)
        lim = max(maximum(pred[keep]), maximum(fit[keep])) * 1.1
        plot!(plt, [0, lim], [0, lim]; ls=:dash, color=:gray, label="y = x")
        savefig(plt, joinpath(results_dir, "fig_rate_vs_rho.pdf"))
        println(tee, "Figure saved to: $(joinpath(results_dir, "fig_rate_vs_rho.pdf"))")
    end

    println(tee, "\nReading: points at or above y=x show the certificate is a valid lower")
    println(tee, "bound on the observed rate; distance above y=x quantifies conservatism.")
    teardown_logging(tee, logpath)
end

main()
