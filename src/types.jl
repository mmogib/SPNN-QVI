# Type definitions for SPNNQVI
#
# Core types for the Scaled Projection Neural Network for QVI.

"""
    QVIProblem

Defines a quasi variational inequality problem:
Find x ∈ 𝔖(x) such that ⟨F(x), y - x⟩ ≥ 0 for all y ∈ 𝔖(x),
where 𝔖(x) = m(x) + S.

Fields:
- `F`:    Operator F: ℝⁿ → ℝⁿ
- `m`:    Translation mapping m: ℝⁿ → ℝⁿ (defines 𝔖(x) = m(x) + S)
- `S`:    Base closed convex set (as a projection function: z ↦ P_S(z))
- `M`:    Fixed SPD metric matrix (n × n)
- `x0`:   Initial point
- `n`:    Dimension
- `name`: Problem identifier
"""
@kwdef struct QVIProblem
    F::Function                     # Operator F(x) → vector
    m::Function                     # Translation m(x) → vector
    proj_S::Function                # Euclidean projection onto base set S
    M::AbstractMatrix               # Fixed SPD matrix (Matrix, Diagonal, or FactorInverse)
    x0::Vector{Float64}            # Initial point
    n::Int                          # Dimension
    name::String = "unnamed"        # Problem name
    # Optional caches (nothing → legacy behavior, M⁻¹ recomputed per projection call).
    # When provided, Minv is the projection metric M⁻¹ (Diagonal or sparse enables
    # the fast projection paths), lb/ub are the box bounds of S, and Mnorm = ‖M‖.
    Minv::Union{Nothing, AbstractMatrix} = nothing
    lb::Union{Nothing, Vector{Float64}} = nothing
    ub::Union{Nothing, Vector{Float64}} = nothing
    Mnorm::Float64 = NaN
end

"""
    FactorInverse(fact, n, nrm)

Lazy representation of A⁻¹ given a factorization `fact` of A: multiplication
`FactorInverse * v` performs `fact \\ v` (never forming the dense inverse).
`nrm` must be ‖A⁻¹‖ = 1/λ_min(A), supplied at construction (used by `opnorm`).
Used as the matrix M = A⁻¹ for large sparse problems (e.g. the fine-grid obstacle).
"""
struct FactorInverse{TF} <: AbstractMatrix{Float64}
    fact::TF
    n::Int
    nrm::Float64
end

Base.size(op::FactorInverse) = (op.n, op.n)
Base.:*(op::FactorInverse, v::AbstractVector) = op.fact \ Vector{Float64}(v)
LinearAlgebra.opnorm(op::FactorInverse) = op.nrm
LinearAlgebra.opnorm(op::FactorInverse, p::Real) = op.nrm
# getindex is required by the AbstractMatrix interface; O(n) per entry — avoid in hot paths.
function Base.getindex(op::FactorInverse, i::Int, j::Int)
    e = zeros(op.n); e[j] = 1.0
    return (op.fact \ e)[i]
end

"""
    SolverConfig

Configuration for the SPNN QVI solver.
"""
@kwdef struct SolverConfig
    T::Float64 = 20.0              # Time horizon
    dt::Float64 = 0.01             # Time step (Forward Euler)
    alpha::Float64 = 0.1           # Step-size parameter α
    lambda::Float64 = 1.0          # Scaling parameter λ
    tol::Float64 = 1e-6            # Residual tolerance for convergence
    maxiter::Int = 100_000         # Maximum iterations
    verbose::Bool = false          # Print progress
end

"""
    SolverResult

Output of the SPNN QVI solver.
"""
@kwdef struct SolverResult
    x_final::Vector{Float64}       # Final iterate
    residual_final::Float64        # Final residual norm
    iterations::Int                # Number of time steps taken
    converged::Bool                # Whether residual < tol
    time_seconds::Float64          # Wall-clock time
    status::Symbol                 # :optimal, :maxiter, :error
    trajectory::Union{Nothing, Matrix{Float64}} = nothing  # Optional trajectory
    residuals::Union{Nothing, Vector{Float64}} = nothing    # Optional residual history
end
