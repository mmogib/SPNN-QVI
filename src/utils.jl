# Utilities for SPNNQVI

"""
    compute_residual(x, prob, alpha)

Compute the projection residual r(x) = (1/α)(x - T(x)) and its norm.
Convenience wrapper using raw problem data.
"""
function compute_residual(x::Vector{Float64}, prob::QVIProblem, alpha::Float64)
    cfg = SolverConfig(alpha=alpha)
    r, rn = residual(x, prob, cfg)
    return r
end

"""
    V_euclid(x, xstar)

Euclidean Lyapunov function V(x) = ½‖x - x̄‖².
"""
V_euclid(x, xstar) = 0.5 * norm(x - xstar)^2

"""
    V_metric(x, xstar, Minv)

Weighted Lyapunov function V_M(x) = ½‖x - x̄‖²_{M⁻¹} = ½(x-x̄)ᵀM⁻¹(x-x̄).
"""
V_metric(x, xstar, Minv) = 0.5 * dot(x - xstar, Minv * (x - xstar))

"""
    time_to_tol(ts, rs, tol)

Find the first time t at which ‖r(t)‖ ≤ tol. Returns Inf if never reached.
"""
function time_to_tol(ts::Vector{Float64}, rs::Vector{Float64}, tol::Float64)
    for (t, r) in zip(ts, rs)
        r ≤ tol && return t
    end
    return Inf
end

"""
    elapsed_str(seconds)

Format elapsed time as a human-readable string.
"""
function elapsed_str(s::Float64)
    s < 60   && return @sprintf("%.0fs", s)
    s < 3600 && return @sprintf("%.1fm", s / 60)
    return @sprintf("%.1fh", s / 3600)
end
