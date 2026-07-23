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

"""
    reference_residual(x, prob, tau) -> Float64

Method-independent reference residual
  R_ref(x) = ‖x - [m(x) + P_S(x - m(x) - τF(x))]‖ / τ
with the EUCLIDEAN projection onto the fixed base set S and a fixed τ.
Identical for every compared method and metric; used as the common success and
stopping criterion in baseline comparisons. τ should be chosen analytically per
problem (τ = 1/K with K a global Euclidean Lipschitz constant of F), never tuned.
"""
function reference_residual(x::AbstractVector, prob::QVIProblem, tau::Float64)
    mx = prob.m(x)
    z = x - mx
    g = x - (mx + prob.proj_S(z - tau * prob.F(x)))
    return norm(g) / tau
end

"""
    reference_residual_lb(x, prob, tau, lb) -> Float64

Cancellation-reduced evaluation of `reference_residual` for lower-bound box sets
S = {z : z ≥ lb} (e.g. the obstacle problem). Componentwise:
  g_i = F_i(x)              if z_i - τF_i(x) ≥ lb_i   (inactive)
  g_i = (z_i - lb_i)/τ      otherwise                  (active)
with z = x - m(x); returns ‖g‖. Avoids subtracting two nearly equal full vectors
when τ is tiny (fine grids: τ = O(n⁻²)).
"""
function reference_residual_lb(x::AbstractVector, prob::QVIProblem,
                               tau::Float64, lb::Vector{Float64})
    mx = prob.m(x)
    Fx = prob.F(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        z_i = x[i] - mx[i]
        g_i = (z_i - tau * Fx[i] >= lb[i]) ? Fx[i] : (z_i - lb[i]) / tau
        s += g_i * g_i
    end
    return sqrt(s)
end

"""
    with_x0(prob, x0) -> QVIProblem

Copy of `prob` with a different initial point (preserves all cached fields).
"""
with_x0(prob::QVIProblem, x0::Vector{Float64}) = QVIProblem(
    F=prob.F, m=prob.m, proj_S=prob.proj_S, M=prob.M, x0=x0, n=prob.n,
    name=prob.name, Minv=prob.Minv, lb=prob.lb, ub=prob.ub, Mnorm=prob.Mnorm)

"""
    norm_M(prob) -> Float64

‖M‖ (spectral norm), using the cached value when available.
"""
norm_M(prob::QVIProblem) = isnan(prob.Mnorm) ? opnorm(Matrix(prob.M)) : prob.Mnorm
