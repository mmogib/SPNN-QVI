# Test problem definitions for SPNNQVI
#
# Each problem defines a QVI: find x ∈ 𝔖(x) s.t. ⟨F(x), y-x⟩ ≥ 0 ∀y ∈ 𝔖(x)
# where 𝔖(x) = m(x) + S.

const PROBLEM_REGISTRY = Dict{Int, Function}()

function get_problem(id::Int; kwargs...)
    haskey(PROBLEM_REGISTRY, id) || error("Problem $id not found. Available: $(sort(collect(keys(PROBLEM_REGISTRY))))")
    return PROBLEM_REGISTRY[id](; kwargs...)
end

function list_problems()
    ids = sort(collect(keys(PROBLEM_REGISTRY)))
    return [(id, get_problem(id).name) for id in ids]
end

# ── Helpers ─────────────────────────────────────────────────────────────

proj_box(z, lb, ub) = clamp.(z, lb, ub)
rot2(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

"""Compute x̄ for the affine QVI: F(x)=Qx+q, m(x)=δx, S=[lb,ub]^n.
The QVI reduces to VI on S with modified operator."""
function solve_affine_qvi_box(Q, q, δ, lb, ub, n)
    # x̄ solves QVI iff z̄ = (1-δ)x̄ solves VI(F̃, S) with F̃(z) = Q z/(1-δ) + q
    # Solve via projection iteration
    z = 0.5 * (lb + ub) * ones(n)
    α_inner = 0.01
    for _ in 1:100_000
        Fz = Q * z / (1 - δ) + q
        z_new = clamp.(z - α_inner * Fz, lb, ub)
        if norm(z_new - z) < 1e-12
            z = z_new
            break
        end
        z = z_new
    end
    xbar = z / (1 - δ)
    return xbar
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 1: 2D affine QVI on [-1,1]², Euclidean metric
# ══════════════════════════════════════════════════════════════════════════
PROBLEM_REGISTRY[1] = function(; κ::Float64=50.0, ϕ::Float64=π/6, δ::Float64=0.1,
                                 metric::Symbol=:identity)
    n = 2
    R = rot2(ϕ)
    Q = Symmetric(R' * Diagonal([κ, 1.0]) * R)
    # Prescribe solution x̄ = (0.5, 0.3)
    xbar = [0.5, 0.3]
    zbar = (1 - δ) * xbar
    q = Q * zbar / (1 - δ)   # so that F(x̄) = Q x̄ + q gives the right VI condition

    # Actually, for x̄ to solve QVI, we need ⟨F(x̄), y - x̄⟩ ≥ 0 ∀y ∈ δx̄ + [-1,1]²
    # With F(x) = Qx - b, set b = Qx̄ so F(x̄) = 0 → x̄ is trivially a solution
    b = Q * xbar
    F(x) = Q * x - b

    m(x) = δ * x
    lb, ub = -ones(n), ones(n)
    proj_S(z) = proj_box(z, lb, ub)

    M = if metric == :identity
        Matrix{Float64}(I, n, n)
    elseif metric == :Qinv
        Matrix{Float64}(inv(Q))       # perfect preconditioner: M·F = x - Q⁻¹b
    elseif metric == :diag_inv
        Diagonal(1.0 ./ [κ, 1.0]) |> Matrix{Float64}  # diagonal approx of Q⁻¹
    elseif metric == :Q
        Matrix{Float64}(Q)
    elseif metric == :diag
        Diagonal([κ, 1.0]) |> Matrix{Float64}
    else
        Matrix{Float64}(I, n, n)
    end

    x0 = [0.9, -0.8]  # corner-ish start

    kappa_str = isinteger(κ) ? string(Int(κ)) : string(κ)
    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="affine_2d_$(metric)_kappa$(kappa_str)_delta$(δ)")
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 2: Same as 1 but with rotated box S_θ = R(θ)[-1,1]²
# ══════════════════════════════════════════════════════════════════════════
PROBLEM_REGISTRY[2] = function(; κ::Float64=50.0, ϕ::Float64=π/6, δ::Float64=0.1,
                                 θset::Float64=π/8, metric::Symbol=:identity)
    n = 2
    R_Q = rot2(ϕ)
    Q = Symmetric(R_Q' * Diagonal([κ, 1.0]) * R_Q)
    xbar = [0.5, 0.3]
    b = Q * xbar
    F(x) = Q * x - b
    m(x) = δ * x

    R_S = rot2(θset)
    lb, ub = -ones(n), ones(n)
    # Projection onto rotated box: rotate, clip, rotate back
    proj_S(z) = R_S * proj_box(R_S' * z, lb, ub)

    M = metric == :identity ? Matrix{Float64}(I, n, n) : Matrix{Float64}(Q)
    x0 = [0.9, -0.8]

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="affine_2d_rotbox_$(metric)")
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 3: 2D nonlinear QVI (interior solution)
# ══════════════════════════════════════════════════════════════════════════
PROBLEM_REGISTRY[3] = function(; κ::Float64=25.0, ϕ_Q::Float64=0.314,
                                 γ::Float64=0.3, c::Float64=2.0,
                                 δ::Float64=0.1, metric::Symbol=:identity)
    n = 2
    R = rot2(ϕ_Q)
    Q = Symmetric(R' * Diagonal([κ, 1.0]) * R)
    xbar = [0.4, -0.3]
    # Choose b so that F(x̄) = 0 (interior solution)
    b = Q * xbar + γ * (exp.(-c * xbar) .- 1)
    F(x) = Q * x - b + γ * (exp.(-c * x) .- 1)

    m(x) = δ * x
    lb, ub = -ones(n), ones(n)
    proj_S(z) = proj_box(z, lb, ub)

    M = metric == :identity ? Matrix{Float64}(I, n, n) : Matrix{Float64}(Q)
    x0 = [0.8, 0.8]

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="nonlinear_2d_$(metric)")
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 10: Higher-dimensional affine QVI
# ══════════════════════════════════════════════════════════════════════════
PROBLEM_REGISTRY[10] = function(; n::Int=10, cond_Q::Float64=100.0,
                                  δ::Float64=0.1, seed::Int=42,
                                  metric::Symbol=:identity)
    rng = MersenneTwister(seed)
    # Random SPD Q with controlled condition number
    U, _ = qr(randn(rng, n, n))
    U = Matrix(U)
    eigs = range(1.0, cond_Q, length=n)
    Q = Symmetric(U' * Diagonal(collect(eigs)) * U)

    xbar = 0.5 * ones(n)  # interior of [0,1]^n
    b = Q * xbar
    F(x) = Q * x - b

    m(x) = δ * x
    lb, ub = zeros(n), ones(n)
    proj_S(z) = proj_box(z, lb, ub)

    M = if metric == :identity
        Matrix{Float64}(I, n, n)
    elseif metric == :diag
        Diagonal(diag(Q)) |> Matrix{Float64}
    elseif metric == :Q
        Matrix{Float64}(Q)
    else
        Matrix{Float64}(I, n, n)
    end

    x0 = 0.1 * ones(n) + 0.8 * rand(rng, n)

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="affine_$(n)d_cond$(Int(cond_Q))_$(metric)")
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 5: Discretized Obstacle QVI (1D membrane)
# ══════════════════════════════════════════════════════════════════════════
#
# Implicit obstacle QVI: 1D elastic membrane over a responsive substrate
# (an illustrative author-constructed model; see the manuscript's Exp 5).
# The substrate deforms in response to the membrane's mean displacement:
#   obstacle height ψ(x) = δ·mean(x) + ψ₀
# This creates a genuine QVI: K(x) = m(x) + S where m(x) = δ·mean(x)·1_n
# and S = {v : v ≥ ψ₀}. The contraction constant is K_m = δ < 1 (exact,
# attained at perturbations proportional to 1_n).
#
# Physical motivation: Noor (1988) cites implicit obstacle boundary value problems
# as the canonical application of the translative QVI K(u) = m(u) + K.
# See also Baiocchi-Capelo (1984) and Kikuchi-Oden (1988).
#
# 1D membrane on [0,1], n interior grid points, h = 1/(n+1).
# F(x) = Ax - f where A = (1/h²) tridiag(-1,2,-1) is the stiffness matrix.
# Base obstacle: ψ₀_i = 0.2·sin(π·xᵢ), S = {x : x ≥ ψ₀}.
# m(x)_i = δ·mean(x) (contraction: constant shift by average).
# ──────────────────────────────────────────────────────────────────────────
PROBLEM_REGISTRY[5] = function(; n::Int=20, δ::Float64=0.1, metric::Symbol=:identity,
                                 sparse_impl::Bool = n > 100)
    sparse_impl && return _obstacle_sparse(n, δ, metric)
    h = 1.0 / (n + 1)
    grid = [i * h for i in 1:n]

    # Stiffness matrix: A = (1/h²) tridiag(-1, 2, -1)
    A = zeros(n, n)
    for i in 1:n
        A[i, i] = 2.0 / h^2
        i > 1 && (A[i, i-1] = -1.0 / h^2)
        i < n && (A[i, i+1] = -1.0 / h^2)
    end
    A_sym = Symmetric(A)

    # Uniform load
    f_load = ones(n)

    # Operator: F(x) = Ax - f
    F(x) = A_sym * x - f_load

    # Base obstacle: ψ₀_i = 0.2·sin(π·xᵢ)
    ψ0 = 0.2 * sin.(π * grid)

    # S = {x : x ≥ ψ₀} — projection clamps each component from below
    proj_S(z) = max.(z, ψ0)

    # Translation: m(x)_i = δ·mean(x) (constant vector; Lipschitz constant K_m = δ)
    m(x) = fill(δ * mean(x), n)

    # Metric
    M = if metric == :identity
        Matrix{Float64}(I, n, n)
    elseif metric == :Ainv
        Matrix{Float64}(inv(A_sym))       # preconditioner: inv(stiffness)
    elseif metric == :jacobi
        Diagonal(h^2 / 2.0 * ones(n)) |> Matrix{Float64}  # diag(A)⁻¹
    else
        Matrix{Float64}(I, n, n)
    end

    # Initial point: midway between obstacle and 1.0
    x0 = 0.5 * (ψ0 .+ 1.0)

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="obstacle_$(n)d_$(metric)_delta$(δ)")
end

"""
Sparse/operator implementation of the obstacle QVI (Problem 5) for fine grids.
Same mathematical problem as the dense variant; the stiffness matrix is stored
sparse, F is an O(n) matvec, and the metric options avoid dense inverses:
  :identity → M = I (Diagonal),      projection = componentwise clamp
  :jacobi   → M = diag(A)⁻¹,         projection = componentwise clamp
  :Ainv     → M = A⁻¹ via a sparse Cholesky factorization (FactorInverse);
              projection metric M⁻¹ = A (sparse) via the primal-dual
              active-set method, verified to KKT tolerance (fail-closed).
Analytic spectral data of A = (1/h²)tridiag(-1,2,-1):
  λ_min = (4/h²)sin²(πh/2),  λ_max = (4/h²)cos²(πh/2).
"""
function _obstacle_sparse(n::Int, δ::Float64, metric::Symbol)
    h = 1.0 / (n + 1)
    grid = [i * h for i in 1:n]

    A = spdiagm(-1 => fill(-1.0 / h^2, n - 1),
                 0 => fill( 2.0 / h^2, n),
                 1 => fill(-1.0 / h^2, n - 1))
    λmin_A = (4.0 / h^2) * sin(π * h / 2)^2
    λmax_A = (4.0 / h^2) * cos(π * h / 2)^2

    f_load = ones(n)
    F(x) = A * x - f_load

    ψ0 = 0.2 * sin.(π * grid)
    proj_S(z) = max.(z, ψ0)
    m(x) = fill(δ * mean(x), n)

    lb = copy(ψ0)
    ub = fill(Inf, n)

    M, Minv, Mnorm = if metric == :identity
        Diagonal(ones(n)), Diagonal(ones(n)), 1.0
    elseif metric == :jacobi
        d = h^2 / 2.0
        Diagonal(fill(d, n)), Diagonal(fill(1.0 / d, n)), d
    elseif metric == :Ainv
        fact = cholesky(Symmetric(A))
        FactorInverse(fact, n, 1.0 / λmin_A), A, 1.0 / λmin_A
    else
        Diagonal(ones(n)), Diagonal(ones(n)), 1.0
    end

    x0 = 0.5 * (ψ0 .+ 1.0)

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
               name="obstacle_sparse_$(n)d_$(metric)_delta$(δ)",
               Minv=Minv, lb=lb, ub=ub, Mnorm=Mnorm)
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 6: GNEP — Cournot Duopoly
# ══════════════════════════════════════════════════════════════════════════
#
# N=2 players, x = (x₁, x₂), inverse demand p(Q) = d - λ_p - ρ·Q, Q = Σxᵢ.
# Player i's profit: πᵢ = p(Q)·xᵢ - cᵢ·xᵢ.
# F = pseudo-gradient (VI convention: F = -∇profit).
# S = {x : 0 ≤ xᵢ ≤ capacity, x₁ + x₂ ≤ total_cap}.
#
# The translation m(x) = δ·x models committed production: a fraction δ of each
# player's current output is pre-committed (e.g. under bilateral contracts) and
# S is read as residual-market quotas — per-firm and aggregate limits on the
# uncommitted output (1-δ)x — so feasibility x ∈ m(x) + S shifts the admissible
# set with the state, making the QVI genuinely quasi. A stylized jointly
# constrained Cournot model (not a player-wise generalized Nash derivation).
# K_m = δ < 1 ensures the translation is a contraction.
# ──────────────────────────────────────────────────────────────────────────
PROBLEM_REGISTRY[6] = function(; d::Float64=20.0, λ_p::Float64=4.0, ρ::Float64=1.0,
                                 c1::Float64=1.0, c2::Float64=2.0,
                                 capacity::Float64=10.0, total_cap::Float64=12.0,
                                 δ::Float64=0.1, metric::Symbol=:identity)
    n = 2

    # F = -∇profit (VI convention)
    # F₁(x) = 2ρx₁ + ρx₂ - (d - λ_p - c₁)
    # F₂(x) = ρx₁ + 2ρx₂ - (d - λ_p - c₂)
    a1 = d - λ_p - c1
    a2 = d - λ_p - c2
    function F(x)
        return [2ρ * x[1] + ρ * x[2] - a1,
                ρ * x[1] + 2ρ * x[2] - a2]
    end

    # Translation
    m(x) = δ * x

    # S = {x : 0 ≤ xᵢ ≤ capacity, x₁+x₂ ≤ total_cap}
    # EXACT Euclidean projection (2-D): if the box clamp satisfies the shared
    # constraint it is the projection; otherwise the capacity face is active
    # and the projection is the clamped foot point on that segment.
    function proj_S(z)
        y = clamp.(z, 0.0, capacity)
        if y[1] + y[2] <= total_cap
            return y
        end
        lo = max(0.0, total_cap - capacity)
        hi = min(capacity, total_cap)
        y1 = clamp((z[1] - z[2] + total_cap) / 2, lo, hi)
        return [y1, total_cap - y1]
    end

    # Metric
    M = Matrix{Float64}(I, n, n)

    # Initial point
    x0 = [1.0, 1.0]

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="gnep_cournot_$(metric)_delta$(δ)")
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 7: Nonlinear Monotone Operator (n=5 by default)
# ══════════════════════════════════════════════════════════════════════════
#
# F(x) = Qx + ρ·φ(x) + q, with φᵢ(x) = arctan(xᵢ - 2).
# Q: SPD matrix with eigenvalues [10,5,2,1,0.5] rotated by a fixed orthogonal matrix.
# x̄ = [1,2,3,2,1] (interior of box [0,5]ⁿ), q chosen so F(x̄) = 0.
# m(x) = δ·x, S = [0,5]ⁿ.
# ──────────────────────────────────────────────────────────────────────────
PROBLEM_REGISTRY[7] = function(; n::Int=5, ρ_nl::Float64=1.0, δ::Float64=0.1,
                                 seed::Int=42, metric::Symbol=:identity)
    rng = MersenneTwister(seed)

    # Eigenvalues: spread from large to small
    eigvals_Q = if n == 5
        [10.0, 5.0, 2.0, 1.0, 0.5]
    else
        # General n: geometric spacing from 10 down to 0.5
        10.0 .^ range(log10(10.0), log10(0.5), length=n)
    end

    # Random orthogonal matrix via QR
    U_mat, _ = qr(randn(rng, n, n))
    U_mat = Matrix(U_mat)
    Q = Symmetric(U_mat' * Diagonal(eigvals_Q) * U_mat)

    # Nonlinear part: φᵢ(x) = arctan(xᵢ - 2)
    φ(x) = atan.(x .- 2.0)

    # Known solution (interior of [0,5]ⁿ)
    xbar = if n == 5
        [1.0, 2.0, 3.0, 2.0, 1.0]
    else
        # General n: points in interior of [0,5]
        2.5 * ones(n)
    end

    # q chosen so F(x̄) = 0
    q = -(Q * xbar + ρ_nl * φ(xbar))

    F(x) = Q * x + ρ_nl * φ(x) + q

    # Translation
    m(x) = δ * x

    # Box constraint [0,5]ⁿ
    lb = zeros(n)
    ub = 5.0 * ones(n)
    proj_S(z) = proj_box(z, lb, ub)

    # Metric (with caches for the diagonal fast path)
    M = if metric == :identity
        Matrix{Float64}(I, n, n)
    elseif metric == :Qinv
        Matrix{Float64}(inv(Q))
    elseif metric == :diag_inv
        Diagonal(1.0 ./ diag(Q)) |> Matrix{Float64}   # Jacobi recipe: diag(sym part)⁻¹
    else
        Matrix{Float64}(I, n, n)
    end
    Minv_c  = metric == :diag_inv ? Diagonal(Vector{Float64}(diag(Q))) : nothing
    Mnorm_c = metric == :diag_inv ? maximum(1.0 ./ diag(Q)) : NaN

    # Initial point: near boundary
    x0 = [4.5, 0.5, 4.5, 0.5, 4.5]
    if n != 5
        x0 = repeat([4.5, 0.5], cld(n, 2))[1:n]
    end

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="nonlinear_monotone_$(n)d_$(metric)_delta$(δ)",
              Minv=Minv_c, lb=lb, ub=ub, Mnorm=Mnorm_c)
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 8: Random High-Dimensional QVI (n=50 by default)
# ══════════════════════════════════════════════════════════════════════════
#
# F(x) = Q_op·x + q with Q_op = D + S_skew (SPD + skew-symmetric → strongly monotone).
# D = NᵀN + 0.1I, S_skew random skew-symmetric.
# x̄ random in interior of [0,10]ⁿ, q = -Q_op·x̄.
# m(x) = δ·x, S = [0,10]ⁿ.
# ──────────────────────────────────────────────────────────────────────────
PROBLEM_REGISTRY[8] = function(; n::Int=50, δ::Float64=0.1, seed::Int=1234,
                                 metric::Symbol=:identity)
    rng = MersenneTwister(seed)

    # SPD part: D = NᵀN + 0.1I
    N_mat = randn(rng, n, n)
    D = N_mat' * N_mat + 0.1 * I(n)
    D_sym = Symmetric(D)

    # Skew-symmetric part
    S_raw = randn(rng, n, n)
    S_skew = S_raw - S_raw'  # skew-symmetric: S_skew' = -S_skew

    # Full operator matrix: positive definite + skew → strongly monotone
    Q_op = D_sym + S_skew

    # Known solution in interior of [0,10]ⁿ
    xbar = 1.0 .+ 8.0 * rand(rng, n)  # in (1,9) ⊂ (0,10)

    # q chosen so F(x̄) = 0
    q = -Q_op * xbar

    F(x) = Q_op * x + q

    # Translation
    m(x) = δ * x

    # Box constraint [0,10]ⁿ
    lb = zeros(n)
    ub = 10.0 * ones(n)
    proj_S(z) = proj_box(z, lb, ub)

    # Symmetric part of Q_op for preconditioning
    D_full = Matrix{Float64}(D_sym)

    # Metric (with caches for the diagonal fast path)
    M = if metric == :identity
        Matrix{Float64}(I, n, n)
    elseif metric == :Dinv
        Matrix{Float64}(inv(D_full))     # inv(symmetric part)
    elseif metric == :diag_inv
        Diagonal(1.0 ./ diag(Q_op)) |> Matrix{Float64}  # diag(Q_op)⁻¹
    else
        Matrix{Float64}(I, n, n)
    end
    Minv_c  = metric == :diag_inv ? Diagonal(Vector{Float64}(diag(Q_op))) : nothing
    Mnorm_c = metric == :diag_inv ? maximum(1.0 ./ diag(Q_op)) : NaN

    # Initial point: random in box
    x0 = 10.0 * rand(rng, n)

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="random_highdim_$(n)d_$(metric)_delta$(δ)",
              Minv=Minv_c, lb=lb, ub=ub, Mnorm=Mnorm_c)
end

# ══════════════════════════════════════════════════════════════════════════
# Problem 9: Nonlinear orthant QVI benchmark (manuscript Experiment 4)
# ══════════════════════════════════════════════════════════════════════════
#
# Attribution RESOLVED (2026-07-22): the example appears in NEITHER Noor 2003
# ("Implicit dynamical systems...", AMC 134) NOR Noor 2007 ("On merit functions
# for quasivariational inequalities", J. Math. Inequal. 1(2) — checked in full).
# The original code comment's attribution was erroneous. The manuscript
# presents this as an author-adapted nonlinear benchmark in the setting of
# Noor's Euclidean QVI dynamics, which is accurate.
#
# F(u) = (u₁ + u₂ + sin(u₁), u₁ + u₂ + sin(u₂))
# K = ℝ²₊ = {x : x ≥ 0} (nonneg. orthant), so S = ℝ²₊, proj_S(z) = max.(z, 0)
# m(u) = u/8 (contraction, K_m = 1/8 = 0.125)
# Solution: ū = (0, 0) [verifiable: F(0)=(0,0), ⟨F(0),y-0⟩ = 0 ≥ 0 ∀y ≥ 0]
# M = I (Euclidean metric)
# NOTE: F is strongly monotone near ū but NOT globally monotone on ℝ²₊
# (symmetrized Jacobian has eigenvalue -1 at (π,π)).
# ──────────────────────────────────────────────────────────────────────────
PROBLEM_REGISTRY[9] = function(;)
    n = 2

    # Operator: F(u) = (u₁ + u₂ + sin(u₁), u₁ + u₂ + sin(u₂))
    function F(u)
        return [u[1] + u[2] + sin(u[1]),
                u[1] + u[2] + sin(u[2])]
    end

    # Translation: m(u) = u/8 (contraction with K_m = 1/8)
    m(u) = u / 8

    # S = ℝ²₊ (nonneg. orthant) — projection clamps from below at 0
    proj_S(z) = max.(z, 0.0)

    # Metric: identity
    M = Matrix{Float64}(I, n, n)

    # Initial point
    x0 = [1.0, 1.0]

    QVIProblem(F=F, m=m, proj_S=proj_S, M=M, x0=x0, n=n,
              name="nonlinear_orthant_benchmark")
end
