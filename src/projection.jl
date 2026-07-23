# Metric projection routines for SPNNQVI
#
# P_{S,M⁻¹}(z) = argmin_{y ∈ S} ½(y-z)ᵀM⁻¹(y-z)
# Translation identity: P_{m(x)+S, M⁻¹}(z) = m(x) + P_{S, M⁻¹}(z - m(x))

"""
    metric_project_box(z, Minv, lb, ub)

Compute P_{S,M⁻¹}(z) where S = {x : lb ≤ x ≤ ub} is a box:
  P_{S,M⁻¹}(z) = argmin_{y ∈ S} ½(y-z)ᵀM⁻¹(y-z),
a box-constrained QP, solved by projected coordinate descent
(identity metrics reduce to the componentwise clamp).
"""
function metric_project_box(z::AbstractVector, Minv::Matrix{Float64},
                            lb::Vector{Float64}, ub::Vector{Float64})
    n = length(z)
    # For M⁻¹ = I, just clamp
    if Minv ≈ I(n)
        return clamp.(z, lb, ub)
    end
    # General case: coordinate descent (fast for small-medium n)
    return metric_project_box_cholesky(z, Minv, lb, ub)
end

"""
    metric_project_box_cholesky(z, Minv, lb, ub; maxiter=100)

Compute P_{S,M⁻¹}(z) for a box via coordinate descent on the KKT system.
Falls back to QP if needed. For 2D problems, this is very fast.
"""
function metric_project_box_cholesky(z::AbstractVector, Minv::Matrix{Float64},
                                     lb::Vector{Float64}, ub::Vector{Float64};
                                     maxiter::Int=200, tol::Float64=1e-12)
    n = length(z)
    # For identity metric, just clamp
    if Minv ≈ I(n)
        return clamp.(z, lb, ub)
    end
    # Coordinate descent on the QP: min ½ yᵀ Minv y - zᵀ Minv y  s.t. lb ≤ y ≤ ub
    # Gradient: Minv * y - Minv * z = Minv * (y - z)
    # For each coordinate i, fix all others and solve the 1D problem.
    y = clamp.(z, lb, ub)  # warm start
    g = Minv * z
    for _ in 1:maxiter
        y_old = copy(y)
        for i in 1:n
            # Optimal y_i with all other y_j fixed:
            # ∂/∂y_i [½ yᵀ Minv y - zᵀ Minv y] = (Minv * y)_i - g_i = 0
            # y_i = (g_i - Σ_{j≠i} Minv[i,j] * y_j) / Minv[i,i]
            s = g[i]
            for j in 1:n
                j == i && continue
                s -= Minv[i, j] * y[j]
            end
            y[i] = clamp(s / Minv[i, i], lb[i], ub[i])
        end
        if norm(y - y_old) < tol
            break
        end
    end
    return y
end

"""
    metric_projection(z, prob::QVIProblem)

Compute P_{S,M⁻¹}(z) using the problem's projection structure.
The problem stores proj_S (Euclidean projection onto S) and M.
For box constraints, uses the coordinate descent method.
"""
function metric_projection(z::AbstractVector, prob::QVIProblem)
    # Fast paths when the projection metric M⁻¹ is cached on the problem.
    if prob.Minv !== nothing
        Minv = prob.Minv
        if Minv isa UniformScaling || Minv isa Diagonal
            # Diagonal metric on a separable (box-type) set: the metric projection
            # coincides with the Euclidean projection, computed componentwise.
            return prob.proj_S(z)
        end
        lb, ub = prob.lb === nothing ? _get_box_bounds(prob) :
                 (prob.lb, prob.ub === nothing ? fill(Inf, prob.n) : prob.ub)
        if Minv isa SparseMatrixCSC
            if all(u -> u == Inf, ub) && _pdas_applicable(Minv)
                # Lower-bound box (obstacle-type) with a symmetric Z-matrix
                # metric (M-matrix once SPD): projection via the primal-dual
                # active-set method, verified to KKT tolerance and fail-closed —
                # coordinate descent stalls when cond(Minv) is large (fine grids).
                return _project_lb_pdas(z, Minv, lb)
            end
            return _project_box_cd_sparse(z, Minv, lb, ub)
        end
        return metric_project_box_cholesky(z, Matrix(Minv), lb, ub)
    end
    # Legacy path (numerics identical to the original implementation).
    # Check if metric is identity BEFORE forming the dense inverse.
    if prob.M ≈ I(prob.n)
        return prob.proj_S(z)
    end
    Minv = inv(prob.M)
    # For box problems, extract bounds from proj_S behavior
    # We use the QP approach as the general fallback
    return metric_project_box(z, Minv, _get_box_bounds(prob)...)
end

"""
    _project_box_cd_sparse(z, Minv, lb, ub; maxiter=200, tol=1e-12)

Coordinate descent (projected Gauss--Seidel) for the box-constrained QP
min ½(y-z)ᵀMinv(y-z) s.t. lb ≤ y ≤ ub, with sparse symmetric Minv.
Each sweep is O(nnz(Minv)); for tridiagonal Minv this is O(n) per sweep.
"""
function _project_box_cd_sparse(z::AbstractVector, Minv::SparseMatrixCSC,
                                lb::Vector{Float64}, ub::Vector{Float64};
                                maxiter::Int=200, tol::Float64=1e-12)
    n = length(z)
    y = clamp.(z, lb, ub)                 # warm start
    g = Minv * z
    rows = rowvals(Minv)
    vals = nonzeros(Minv)
    for _ in 1:maxiter
        delta = 0.0
        for i in 1:n
            s = g[i]
            dii = 0.0
            # Column i of the symmetric Minv equals row i.
            for k in nzrange(Minv, i)
                j = rows[k]
                v = vals[k]
                if j == i
                    dii = v
                else
                    s -= v * y[j]
                end
            end
            y_new = clamp(s / dii, lb[i], ub[i])
            delta = max(delta, abs(y_new - y[i]))
            y[i] = y_new
        end
        delta < tol && break
    end
    return y
end

"""
    metric_projection_translated(z, mx, prob::QVIProblem)

Compute P_{m(x)+S, M⁻¹}(z) = m(x) + P_{S, M⁻¹}(z - m(x))
"""
function metric_projection_translated(z::AbstractVector, mx::AbstractVector,
                                      prob::QVIProblem)
    return mx + metric_projection(z - mx, prob)
end

# Diagnostic counters, read/reset by experiment scripts:
#   PDAS_ITER_COUNT   — total inner iterations across all projection calls
#   PDAS_CAP_ACCEPTS  — projections accepted on the cap path (KKT-verified after
#                       hitting the iteration cap); any nonzero value warrants
#                       investigation of the classification behavior
#   PDAS_MAX_KKT      — maximum normalized KKT residual ratio (residual/tolerance)
#                       over accepted projections since last reset
const PDAS_ITER_COUNT = Ref{Int}(0)
const PDAS_CAP_ACCEPTS = Ref{Int}(0)
const PDAS_MAX_KKT = Ref{Float64}(0.0)

const _PDAS_STRUCTURE = IdDict{Any,Bool}()

"""
    _pdas_applicable(Q)

Memoized structural gate for the PDAS fast path (checked once per matrix
instance): `Q` must be symmetric, every diagonal entry must be explicitly
stored and positive, and all off-diagonal stored entries must be nonpositive —
a symmetric Z-matrix. This is a PARTIAL gate: it does not verify positive
definiteness (SPD holds by construction for every projection metric this
package builds); only together with SPD does the Z-structure give the
M-matrix hypothesis of the PDAS global-convergence theory.
"""
function _pdas_applicable(Q::SparseMatrixCSC)
    get!(_PDAS_STRUCTURE, Q) do
        issymmetric(Q) || return false
        rows = rowvals(Q); vals = nonzeros(Q)
        ndiag = 0
        for j in 1:size(Q, 2)
            for k in nzrange(Q, j)
                if rows[k] == j
                    vals[k] > 0 || return false
                    ndiag += 1
                else
                    vals[k] <= 0 || return false
                end
            end
        end
        return ndiag == size(Q, 1)
    end
end

# Scale-aware KKT verification for the lower-bound projection: primal
# feasibility, stationarity on the inactive set, dual feasibility on the
# active set, and componentwise complementarity, with tolerances relative to
# the magnitude of Q and of the data.
function _pdas_kkt_check(y::Vector{Float64}, μ::Vector{Float64},
                         lb::Vector{Float64}, Q::SparseMatrixCSC, z::AbstractVector)
    sQ = maximum(abs, nonzeros(Q))
    zs = max(1.0, maximum(abs, z), maximum(abs, lb))
    tolP = 1e-11 * zs
    tolD = 1e-11 * sQ * zs
    gap = y .- lb
    pinf = -min(0.0, minimum(gap))
    inact = gap .> tolP
    sinf = maximum(abs.(μ[inact]); init=0.0)
    dinf = -min(0.0, minimum(μ[.!inact]; init=0.0))
    cinf = maximum(abs.(μ .* gap); init=0.0)
    ok = (pinf <= tolP) && (sinf <= tolD) && (dinf <= tolD) && (cinf <= tolD * zs)
    ratio = max(pinf / tolP, sinf / tolD, dinf / tolD, cinf / (tolD * zs))
    ok && (PDAS_MAX_KKT[] = max(PDAS_MAX_KKT[], ratio))
    return ok, (pinf=pinf, sinf=sinf, dinf=dinf, cinf=cinf)
end

"""
    _project_lb_pdas_info(z, Q, lb; maxiter=0)

Metric projection onto the lower-bound box {y : y ≥ lb} in the metric Q
(sparse SPD M-matrix): min ½(y-z)ᵀQ(y-z) s.t. y ≥ lb, by the primal-dual
active-set (semismooth Newton) method of Hintermüller–Ito–Kunisch, globally
convergent for M-matrices (e.g. the discrete Laplacian). Each iteration solves
one reduced sparse SPD system; for tridiagonal Q this is O(n). The primal-dual
update uses the HIK strategy with parameter c = max|Q| so the dual and primal
terms are compared on commensurate scales (valid for any fixed c > 0; c = 1
lets floating-point noise in the multipliers, which carry the factor max|Q|,
flip the classification of near-degenerate free-boundary components). The
iteration terminates when the active set stabilizes (typically < 10
iterations) or repeats a 2-cycle, and the accepted iterate is then verified
against the KKT system to a scale-aware tolerance. If the iteration reaches
the cap, the final iterate is accepted if and only if it passes the same KKT
verification (still fail-closed). A cap hit has two distinct causes:
classification chatter at the floating-point noise floor around an
already-converged point (possible at any cap), and genuinely incomplete
active-set peeling under an explicitly undersized user cap (the n=500
regression with `maxiter=50` is a counterexample of the second kind). The
default cap `max(100, n+10)` (when `maxiter <= 0`) is an ENGINEERING cap:
Hintermüller--Ito--Kunisch supply the primal-dual active-set / semismooth
Newton framework and its global convergence for M-matrices; the explicit
(n+1)-iteration bound is associated with the related T-monotone /
policy-iteration formulations (e.g. He and Yang, 2019) under exact reduced
solves and a nonsingular M-matrix. Neither statement is an unconditional
floating-point bound — roundoff chatter and the partial runtime structural
gate lie outside the theory — so the cap is backed by targeted regressions
(near-lb noise-band inputs; the deep-peeling first-KM input at n=500).
Far-from-solution inputs legitimately need O(n) iterations, so no constant
cap suffices; with one O(n) reduced tridiagonal solve per iteration, the
n-scaled cap bounds a single projection by O(n^2) work in the worst case.
Returns `(y, iters, reason, converged, kkt_ok, kkt)` where `reason` is
`:stable` (active set repeated), `:cycle` (2-cycle detected), or `:cap`
(iteration cap reached; accepted only if KKT passes), and `converged` means
`reason !== :cap`.
"""
function _project_lb_pdas_info(z::AbstractVector, Q::SparseMatrixCSC,
                               lb::Vector{Float64}; maxiter::Int=0)
    n = length(z)
    maxiter = maxiter <= 0 ? max(100, n + 10) : maxiter
    zv = Vector{Float64}(z)
    Qz = Q * zv
    sQ = maximum(abs, nonzeros(Q))            # classification scale (HIK c = sQ)
    y = Vector{Float64}(max.(zv, lb))         # warm start: clamp
    μ = Q * (y - zv)
    active = BitVector(zv .<= lb)             # initial active guess
    prev = falses(0)                          # cycle guard (2-cycles)
    it = 0
    while it < maxiter
        it += 1
        PDAS_ITER_COUNT[] += 1
        I_idx = findall(.!active)
        A_idx = findall(active)
        y_new = Vector{Float64}(undef, n)
        @views y_new[A_idx] .= lb[A_idx]
        if !isempty(I_idx)
            QII = Q[I_idx, I_idx]
            rhs = Qz[I_idx]
            if !isempty(A_idx)
                rhs -= Q[I_idx, A_idx] * lb[A_idx]
            end
            y_new[I_idx] = cholesky(Symmetric(QII)) \ rhs
        end
        # Multipliers: μ = Q(y - z) (zero on inactive up to the solve residual)
        μ = Q * (y_new - zv)
        # Scale-aware primal-dual update: active iff μ_i - sQ·(y_i - lb_i) > 0
        active_new = BitVector((μ .- sQ .* (y_new .- lb)) .> 0)
        y = y_new
        if active_new == active || active_new == prev
            reason = active_new == active ? :stable : :cycle
            kkt_ok, kkt = _pdas_kkt_check(y, μ, lb, Q, zv)
            return (y=y, iters=it, reason=reason, converged=true, kkt_ok=kkt_ok, kkt=kkt)
        end
        prev = active
        active = active_new
    end
    # Iteration cap: verify the final iterate before deciding — accept iff it
    # passes the KKT check (noise-floor chatter around a converged point).
    kkt_ok, kkt = _pdas_kkt_check(y, μ, lb, Q, zv)
    kkt_ok && (PDAS_CAP_ACCEPTS[] += 1)
    return (y=y, iters=it, reason=:cap, converged=false, kkt_ok=kkt_ok, kkt=kkt)
end

"""
    _project_lb_pdas(z, Q, lb; maxiter=0)

Fail-closed wrapper around [`_project_lb_pdas_info`](@ref); the default
`maxiter <= 0` resolves to `max(100, n+10)` exactly as there. Returns the
projection only if the accepted iterate passed the KKT verification;
otherwise throws. An unverified projection is never returned silently.
"""
function _project_lb_pdas(z::AbstractVector, Q::SparseMatrixCSC,
                          lb::Vector{Float64}; maxiter::Int=0)
    out = _project_lb_pdas_info(z, Q, lb; maxiter=maxiter)
    out.kkt_ok || error("PDAS projection failed the KKT verification after " *
        "$(out.iters) iterations" *
        (out.converged ? "" : " (iteration cap reached)") *
        ": pinf=$(out.kkt.pinf), sinf=$(out.kkt.sinf), dinf=$(out.kkt.dinf), " *
        "cinf=$(out.kkt.cinf); failing closed.")
    return out.y
end

# Internal: extract box bounds from a QVIProblem
# Uses the cached lb/ub fields when set; otherwise infers from proj_S.
function _get_box_bounds(prob::QVIProblem)
    if prob.lb !== nothing && prob.ub !== nothing
        return prob.lb, prob.ub
    end
    # Infer from proj_S: project very large/small vectors
    lb = prob.proj_S(fill(-1e10, prob.n))
    ub = prob.proj_S(fill(1e10, prob.n))
    return lb, ub
end
