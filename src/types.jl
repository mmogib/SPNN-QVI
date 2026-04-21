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
    M::Matrix{Float64}              # Fixed SPD metric matrix
    x0::Vector{Float64}            # Initial point
    n::Int                          # Dimension
    name::String = "unnamed"        # Problem name
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
