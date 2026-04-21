# ============================================================================
# s70: Figure Generation for SPNN-QVI Paper
# ============================================================================
#
# Goal:   Generate publication-quality figures via PGFPlotsX (native LaTeX)
# Input:  results/{example1, alpha_sweep, beyond_theory}/
# Output: results/figures/fig_*.pdf
#
# Usage:
#   cd jcode
#   julia --project=. scripts/s70_figures.jl
# ============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using SPNNQVI
using Printf, DelimitedFiles
using Plots; pgfplotsx()
using LaTeXStrings

# ── PGFPlotsX preamble: load amsmath for \text{} ─────────────────────────
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")

# ── Paths ──────────────────────────────────────────────────────────────────

const RESULTS = joinpath(@__DIR__, "..", "results")
const OUTPUT  = joinpath(RESULTS, "figures")
mkpath(OUTPUT)

# ── Helpers ────────────────────────────────────────────────────────────────

"""Read a trajectory CSV (t,x1,x2,residual,V) and return named columns."""
function read_traj(path)
    data = readdlm(path, ','; header=true)[1]
    return (t=data[:,1], x1=data[:,2], x2=data[:,3], r=data[:,4], V=data[:,5])
end

"""Read a generic CSV with header."""
function read_csv(path)
    raw = readdlm(path, ','; header=true)
    return raw[1], vec(raw[2])
end

"""Downsample a vector to at most n_max points, keeping first and last."""
function downsample(x, n_max)
    n = length(x)
    n <= n_max && return x
    idx = unique(round.(Int, range(1, n, length=n_max)))
    return x[idx]
end

"""Downsample multiple vectors in lockstep."""
function downsample_vecs(n_max, vecs...)
    n = length(first(vecs))
    n <= n_max && return vecs
    idx = unique(round.(Int, range(1, n, length=n_max)))
    return Tuple(v[idx] for v in vecs)
end

# Style constants
const METRIC_COLORS = Dict(
    :identity => :blue,
    :Qinv     => :red,
    :diag_inv => RGB(0.0, 0.6, 0.0),  # dark green
)
const METRIC_LABELS = Dict(
    :identity => L"M = I",
    :Qinv     => L"M = Q^{-1}",
    :diag_inv => L"M = \mathrm{diag}(1/\kappa, 1)",
)
const METRIC_STYLES = Dict(
    :identity => :solid,
    :Qinv     => :dash,
    :diag_inv => :dot,
)

const FONT_TITLE  = 10
const FONT_GUIDE  = 9
const FONT_TICK   = 8
const FONT_LEGEND = 7
const LW = 1.5

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Phase Portrait — Metric Reshapes Trajectories
# ══════════════════════════════════════════════════════════════════════════

println("Figure 1: Phase portrait...")

panels_1 = []
metrics_fig1 = [:identity, :Qinv, :diag_inv]
xstar = [0.5, 0.3]

for met in metrics_fig1
    p = plot(;
        xlabel = L"x_1", ylabel = L"x_2",
        title  = METRIC_LABELS[met],
        titlefontsize  = FONT_TITLE,
        guidefontsize  = FONT_GUIDE,
        tickfontsize   = FONT_TICK,
        legendfontsize = FONT_LEGEND,
        legend = :none,
        aspect_ratio = :equal,
        xlims = (-1.15, 1.15), ylims = (-1.15, 1.15),
    )

    # Draw constraint box [-1,1]²
    box_x = [-1, 1, 1, -1, -1]
    box_y = [-1, -1, 1, 1, -1]
    plot!(p, box_x, box_y; lc=:gray, ls=:solid, lw=0.8, label="")

    # Plot trajectories
    for idx in 1:8
        traj_file = joinpath(RESULTS, "example1", "traj_$(met)_x0_$(idx).csv")
        if !isfile(traj_file)
            continue
        end
        tr = read_traj(traj_file)
        # Downsample to avoid overly dense paths
        t_ds, x1_ds, x2_ds = downsample_vecs(200, tr.t, tr.x1, tr.x2)
        plot!(p, x1_ds, x2_ds;
            lc = METRIC_COLORS[met], lw = 0.8, alpha = 0.7, label = "")
        # Mark initial point
        scatter!(p, [tr.x1[1]], [tr.x2[1]];
            mc = METRIC_COLORS[met], ms = 3, msw = 0, label = "")
    end

    # Mark solution
    scatter!(p, [xstar[1]], [xstar[2]];
        mc = :black, shape = :star5, ms = 6, msw = 0, label = "")

    push!(panels_1, p)
end

fig1 = plot(panels_1...;
    layout = (1, 3),
    size   = (1050, 350),
    margin = 5Plots.mm,
)
savefig(fig1, joinpath(OUTPUT, "fig1_phase_portrait.pdf"))
println("  Saved: fig1_phase_portrait.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Residual + Lyapunov Decay Comparison
# ══════════════════════════════════════════════════════════════════════════

println("Figure 2: Residual + Lyapunov decay...")

# Use x0 #1 = [1.0, 1.0] for all metrics
p_res = plot(;
    xlabel = L"t", ylabel = L"\|r(x(t))\|",
    yscale = :log10,
    ylims  = (1e-8, 1e2),
    titlefontsize  = FONT_TITLE,
    guidefontsize  = FONT_GUIDE,
    tickfontsize   = FONT_TICK,
    legendfontsize = FONT_LEGEND,
    legend = :topright,
)

p_lyap = plot(;
    xlabel = L"t", ylabel = L"V(x(t))",
    yscale = :log10,
    ylims  = (1e-16, 1e1),
    titlefontsize  = FONT_TITLE,
    guidefontsize  = FONT_GUIDE,
    tickfontsize   = FONT_TICK,
    legendfontsize = FONT_LEGEND,
    legend = :topright,
)

for met in metrics_fig1
    traj_file = joinpath(RESULTS, "example1", "traj_$(met)_x0_1.csv")
    if !isfile(traj_file)
        continue
    end
    tr = read_traj(traj_file)

    # Filter positive values for log scale and clamp to floor to avoid noise
    r_floor = 1e-8
    V_floor = 1e-16

    r_mask = tr.r .> r_floor
    V_mask = tr.V .> V_floor

    # Downsample to avoid dense oscillation artifacts
    t_r, r_r = downsample_vecs(300, tr.t[r_mask], tr.r[r_mask])
    t_V, V_V = downsample_vecs(300, tr.t[V_mask], tr.V[V_mask])

    plot!(p_res, t_r, r_r;
        label = METRIC_LABELS[met],
        lc = METRIC_COLORS[met],
        ls = METRIC_STYLES[met],
        lw = LW,
    )
    plot!(p_lyap, t_V, V_V;
        label = METRIC_LABELS[met],
        lc = METRIC_COLORS[met],
        ls = METRIC_STYLES[met],
        lw = LW,
    )
end

fig2 = plot(p_res, p_lyap;
    layout = (1, 2),
    size   = (700, 300),
    margin = 5Plots.mm,
)
savefig(fig2, joinpath(OUTPUT, "fig2_decay.pdf"))
println("  Saved: fig2_decay.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Step-size α — Theory vs Practice
# ══════════════════════════════════════════════════════════════════════════

println("Figure 3: Step-size sweep...")

p3 = plot(;
    xlabel = L"\alpha",
    ylabel = L"t(\|r\| \leq 10^{-6})",
    xscale = :log10, yscale = :log10,
    yticks = ([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"]),
    titlefontsize  = FONT_TITLE,
    guidefontsize  = FONT_GUIDE,
    tickfontsize   = FONT_TICK,
    legendfontsize = FONT_LEGEND,
    legend = :outertopright,
)

for (met, label) in [(:identity, L"M = I"), (:Qinv, L"M = Q^{-1}")]
    sweep_file = joinpath(RESULTS, "alpha_sweep", "sweep_$(met).csv")
    if !isfile(sweep_file)
        continue
    end
    data, hdr = read_csv(sweep_file)
    # Columns: alpha, alpha_M_K, r_final, V_final, t_tol2, t_tol6, converged, diverged
    alphas = data[:, 1]
    t6     = data[:, 6]

    # Filter finite values
    mask = isfinite.(t6) .& (t6 .> 0)
    plot!(p3, alphas[mask], t6[mask];
        label = label,
        lc    = METRIC_COLORS[met],
        ls    = METRIC_STYLES[met],
        lw    = LW,
        marker = :circle, ms = 3, msw = 0,
    )
end

# Theoretical bound
vline!(p3, [0.018]; lc=:gray, ls=:dashdot, lw=1.0,
    label=L"\alpha_{\text{theory}} = 0.018")

fig3 = plot(p3;
    size   = (400, 300),
    margin = 5Plots.mm,
)
savefig(fig3, joinpath(OUTPUT, "fig3_alpha_sweep.pdf"))
println("  Saved: fig3_alpha_sweep.pdf")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Monotonicity–Contraction Heatmap
# ══════════════════════════════════════════════════════════════════════════

println("Figure 4: Monotonicity–contraction heatmap...")

begin
    csv_file = joinpath(RESULTS, "beyond_theory", "test3_nonmonotone.csv")
    data, hdr = read_csv(csv_file)
    # Columns: epsilon, delta, r_final, V_final, status
    epsilons_raw = data[:, 1]
    deltas_raw   = data[:, 2]
    r_finals     = data[:, 3]

    # Build grid
    eps_vals   = sort(unique(epsilons_raw), rev=true)   # descending for y-axis
    delta_vals = sort(unique(deltas_raw))
    n_eps = length(eps_vals)
    n_del = length(delta_vals)

    Z = fill(NaN, n_eps, n_del)
    for k in eachindex(epsilons_raw)
        i = findfirst(==(epsilons_raw[k]), eps_vals)
        j = findfirst(==(deltas_raw[k]), delta_vals)
        Z[i, j] = log10(max(r_finals[k], 1e-16))
    end

    p4 = heatmap(
        string.(delta_vals), string.(eps_vals), Z;
        xlabel = L"\delta = K_m",
        ylabel = L"\varepsilon \;\; (\text{monotonicity})",
        colorbar_title = L"\log_{10} \|r\|_{\text{final}}",
        color  = cgrad(:RdYlGn, rev=true),
        clims  = (-7, 1),
        titlefontsize  = FONT_TITLE,
        guidefontsize  = FONT_GUIDE,
        tickfontsize   = FONT_TICK,
        legendfontsize = FONT_LEGEND,
    )

    fig4 = plot(p4;
        size   = (450, 350),
        margin = 5Plots.mm,
    )
    savefig(fig4, joinpath(OUTPUT, "fig4_heatmap.pdf"))
    println("  Saved: fig4_heatmap.pdf")
end


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Extreme α — Convergence Time vs α
# ══════════════════════════════════════════════════════════════════════════

println("Figure 5: Extreme α — convergence time...")

# Instead of plotting noisy r_final (all ~1e-6), plot convergence TIME vs α
# which shows a clear monotonic trend. Extract from logs or use t(1e-6) data.
# Since the CSV only has r_final, we plot a bar-style comparison.

begin
    csv_file = joinpath(RESULTS, "beyond_theory", "test4_extreme_alpha.csv")
    data, hdr = read_csv(csv_file)
    # Columns: alpha, delta, r_final, V_final, status
    alphas_raw = data[:, 1]
    deltas_raw = data[:, 2]
    r_finals   = data[:, 3]

    p5 = plot(;
        xlabel = L"\alpha",
        ylabel = L"\|r\|_{\text{final}}",
        xscale = :log10, yscale = :log10,
        ylims  = (1e-8, 1e-5),
        yticks = ([1e-8, 1e-7, 1e-6, 1e-5],
                  [L"10^{-8}", L"10^{-7}", L"10^{-6}", L"10^{-5}"]),
        titlefontsize  = FONT_TITLE,
        guidefontsize  = FONT_GUIDE,
        tickfontsize   = FONT_TICK,
        legendfontsize = FONT_LEGEND,
        legend = :topright,
    )

    delta_colors = Dict(0.0 => :blue, 0.1 => :red, 0.5 => RGB(0.0, 0.6, 0.0))
    delta_styles = Dict(0.0 => :solid, 0.1 => :dash, 0.5 => :dot)

    for δ in [0.0, 0.1, 0.5]
        mask = deltas_raw .== δ
        αs = alphas_raw[mask]
        rs = r_finals[mask]
        sort_idx = sortperm(αs)

        plot!(p5, αs[sort_idx], rs[sort_idx];
            label  = latexstring(@sprintf("\\delta = %.1f", δ)),
            lc     = delta_colors[δ],
            ls     = delta_styles[δ],
            lw     = LW,
            marker = :circle, ms = 3, msw = 0,
        )
    end

    # Reference line at tol = 1e-6
    hline!(p5, [1e-6]; lc=:gray, ls=:dashdot, lw=0.8,
        label=L"\text{tol} = 10^{-6}")

    fig5 = plot(p5;
        size   = (400, 300),
        margin = 5Plots.mm,
    )
    savefig(fig5, joinpath(OUTPUT, "fig5_extreme_alpha.pdf"))
    println("  Saved: fig5_extreme_alpha.pdf")
end


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Noor (2003) Phase Portrait — Trajectories on R²₊
# ══════════════════════════════════════════════════════════════════════════

println("Figure 6: Noor phase portrait...")

begin
    p6 = plot(;
        xlabel = L"x_1", ylabel = L"x_2",
        titlefontsize  = FONT_TITLE,
        guidefontsize  = FONT_GUIDE,
        tickfontsize   = FONT_TICK,
        legendfontsize = FONT_LEGEND,
        legend = :none,
        aspect_ratio = :equal,
    )

    # Draw R²₊ boundary (axes)
    plot!(p6, [0, 6], [0, 0]; lc=:gray, ls=:solid, lw=0.8, label="")
    plot!(p6, [0, 0], [0, 6]; lc=:gray, ls=:solid, lw=0.8, label="")

    traj_colors = [:blue, :red, RGB(0.0,0.6,0.0), :orange, :purple, :brown]

    for idx in 1:6
        traj_file = joinpath(RESULTS, "noor_comparison", "traj_x0_$(idx).csv")
        if !isfile(traj_file)
            continue
        end
        local data = readdlm(traj_file, ','; header=true)[1]
        local t = data[:, 1]
        local x1 = data[:, 2]
        local x2 = data[:, 3]

        # Downsample
        t_ds, x1_ds, x2_ds = downsample_vecs(200, t, x1, x2)
        plot!(p6, x1_ds, x2_ds;
            lc = traj_colors[idx], lw = 1.0, alpha = 0.8, label = "")
        # Mark initial point
        scatter!(p6, [x1[1]], [x2[1]];
            mc = traj_colors[idx], ms = 4, msw = 0, label = "")
    end

    # Mark solution at origin
    scatter!(p6, [0.0], [0.0];
        mc = :black, shape = :star5, ms = 7, msw = 0, label = "")

    fig6 = plot(p6;
        size   = (380, 350),
        margin = 5Plots.mm,
    )
    savefig(fig6, joinpath(OUTPUT, "fig6_noor_phase.pdf"))
    println("  Saved: fig6_noor_phase.pdf")
end


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Obstacle Membrane Profile — Solution Settling onto Obstacle
# ══════════════════════════════════════════════════════════════════════════

println("Figure 7: Obstacle membrane profile...")

begin
    # Read the converged profile (using Ainv metric which converges)
    profile_file = joinpath(RESULTS, "obstacle", "profile_Ainv.csv")
    prof_data = readdlm(profile_file, ','; header=true)[1]
    grid_x = prof_data[:, 1]
    x_final = prof_data[:, 2]
    obstacle = prof_data[:, 3]

    # Read the trajectory to get snapshots at different times
    traj_file = joinpath(RESULTS, "obstacle", "traj_Ainv_x0_1.csv")
    traj_data = readdlm(traj_file, ','; header=true)[1]
    t_traj = traj_data[:, 1]
    # Trajectory has selected components (x1, x5, x10, x15, x20) — not full profile
    # Instead, reconstruct from initial condition and final profile

    n_grid = length(grid_x)

    p7 = plot(;
        xlabel = L"x", ylabel = L"u(x)",
        titlefontsize  = FONT_TITLE,
        guidefontsize  = FONT_GUIDE,
        tickfontsize   = FONT_TICK,
        legendfontsize = FONT_LEGEND,
        legend = :outertopright,
        xlims  = (0, 1),
    )

    # Plot obstacle (base)
    plot!(p7, grid_x, obstacle;
        lc = :black, ls = :dash, lw = 1.5,
        label = L"\psi_0(x)",
        fillrange = zeros(n_grid), fillalpha = 0.1, fillcolor = :gray,
    )

    # Plot initial membrane (midway between obstacle and 1.0)
    x0_membrane = 0.5 * (obstacle .+ 1.0)
    plot!(p7, grid_x, x0_membrane;
        lc = :blue, ls = :dot, lw = 1.0, alpha = 0.6,
        label = L"u(x,0)",
    )

    # Plot converged membrane
    plot!(p7, grid_x, x_final;
        lc = :red, ls = :solid, lw = 2.0,
        label = L"u(x,T)",
    )

    fig7 = plot(p7;
        size   = (450, 300),
        margin = 5Plots.mm,
    )
    savefig(fig7, joinpath(OUTPUT, "fig7_obstacle_profile.pdf"))
    println("  Saved: fig7_obstacle_profile.pdf")
end


# ══════════════════════════════════════════════════════════════════════════

println("\nAll figures saved to: $OUTPUT")
println("Copy to paper/imgs/ when ready.")
