# Metric projection routines for SPNNQVI
#
# P_{S,M⁻¹}(z) = argmin_{y ∈ S} ½(y-z)ᵀM⁻¹(y-z)
# Translation identity: P_{m(x)+S, M⁻¹}(z) = m(x) + P_{S, M⁻¹}(z - m(x))

"""
    metric_project_box(z, Minv, lb, ub)

Compute P_{S,M⁻¹}(z) where S = {x : lb ≤ x ≤ ub} is a box.

Uses the Cholesky transformation: if M⁻¹ = LLᵀ, then
  P_{S,M⁻¹}(z) = argmin_{y ∈ S} ½(y-z)ᵀM⁻¹(y-z)
which is a box-constrained QP. We solve it via JuMP/HiGHS.
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
    Minv = inv(prob.M)
    # Check if metric is identity
    if prob.M ≈ I(prob.n)
        return prob.proj_S(z)
    end
    # For box problems, extract bounds from proj_S behavior
    # We use the QP approach as the general fallback
    return metric_project_box(z, Minv, _get_box_bounds(prob)...)
end

"""
    metric_projection_translated(z, mx, prob::QVIProblem)

Compute P_{m(x)+S, M⁻¹}(z) = m(x) + P_{S, M⁻¹}(z - m(x))
"""
function metric_projection_translated(z::AbstractVector, mx::AbstractVector,
                                      prob::QVIProblem)
    return mx + metric_projection(z - mx, prob)
end

# Internal: extract box bounds from a QVIProblem
# Convention: problems store lb, ub as fields or we infer from proj_S
function _get_box_bounds(prob::QVIProblem)
    # Default: [0,1]^n — override in problem definition if different
    if hasproperty(prob, :lb) && hasproperty(prob, :ub)
        return prob.lb, prob.ub
    end
    # Try to infer from proj_S: project very large/small vectors
    lb = prob.proj_S(fill(-1e10, prob.n))
    ub = prob.proj_S(fill(1e10, prob.n))
    return lb, ub
end
