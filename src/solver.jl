# Core SPNN-QVI solver
#
# Integrates: dx/dt = λ[P_{𝔖(x),M⁻¹}(x - αMF(x)) - x]
# using Forward Euler with optional trajectory recording.

"""
    T_map(x, prob, cfg) -> y

Evaluate the fixed-point map T(x) = m(x) + P_{S,M⁻¹}(x - m(x) - αMF(x)).
"""
function T_map(x::AbstractVector, prob::QVIProblem, cfg::SolverConfig)
    mx = prob.m(x)
    Fx = prob.F(x)
    # Legacy order (α·M)·Fx for dense M (bit-identical to the original code);
    # operator-safe order α·(M·Fx) for Diagonal / FactorInverse / sparse M,
    # where forming α·M densely would be wasteful or defeat the factorization.
    MFx = prob.M isa Matrix ? cfg.alpha * prob.M * Fx : cfg.alpha * (prob.M * Fx)
    z = x - mx - MFx                        # argument to P_{S,M⁻¹}
    p = metric_projection(z, prob)          # P_{S,M⁻¹}(z)
    return mx + p
end

"""
    residual(x, prob, cfg) -> (r, rnorm)

Compute the projection residual r(x) = (1/α)(x - T(x)) and its norm.
"""
function residual(x::Vector{Float64}, prob::QVIProblem, cfg::SolverConfig)
    Tx = T_map(x, prob, cfg)
    r = (x - Tx) / cfg.alpha
    return r, norm(r)
end

"""
    solve_qvi(prob, cfg; save_trajectory=false) -> SolverResult

Integrate the SPNN-QVI dynamics via Forward Euler:
  x_{k+1} = x_k + dt * λ * (T(x_k) - x_k)

Stops when ‖r(x)‖ < tol or iteration limit is reached.
"""
function solve_qvi(prob::QVIProblem, cfg::SolverConfig;
                   save_trajectory::Bool=false,
                   save_every::Int=1)
    t_start = time()
    x = copy(prob.x0)
    n = prob.n
    dt = cfg.dt
    λ = cfg.lambda

    # Trajectory storage
    if save_trajectory
        max_save = div(cfg.maxiter, save_every) + 1
        traj = zeros(n, max_save)
        res_hist = zeros(max_save)
        t_hist = zeros(max_save)
        traj[:, 1] = x
        _, rn = residual(x, prob, cfg)
        res_hist[1] = rn
        t_hist[1] = 0.0
        save_idx = 1
    end

    converged = false
    k = 0
    rnorm = Inf
    t_current = 0.0

    for iter in 1:cfg.maxiter
        k = iter
        Tx = T_map(x, prob, cfg)
        x_new = x + dt * λ * (Tx - x)
        x = x_new
        t_current += dt

        # Compute residual
        _, rnorm = residual(x, prob, cfg)

        # Save trajectory
        if save_trajectory && (iter % save_every == 0)
            save_idx += 1
            if save_idx <= max_save
                traj[:, save_idx] = x
                res_hist[save_idx] = rnorm
                t_hist[save_idx] = t_current
            end
        end

        # Check convergence
        if rnorm < cfg.tol
            converged = true
            break
        end

        # Check divergence
        if any(isnan, x) || any(isinf, x) || norm(x) > 1e15
            break
        end

        # Verbose output
        if cfg.verbose && (iter % 1000 == 0)
            @printf("  iter %6d  t=%.3f  ‖r‖=%.4e\n", iter, t_current, rnorm)
        end
    end

    elapsed = time() - t_start
    status = converged ? :optimal : (any(isnan, x) ? :error : :maxiter)

    # Trim trajectory
    traj_out = nothing
    res_out = nothing
    if save_trajectory
        traj_out = traj[:, 1:save_idx]
        res_out = res_hist[1:save_idx]
    end

    return SolverResult(
        x_final = x,
        residual_final = rnorm,
        iterations = k,
        converged = converged,
        time_seconds = elapsed,
        status = status,
        trajectory = traj_out,
        residuals = res_out
    )
end

"""
    solve_qvi_diffeq(prob, cfg; solver=Tsit5(), save_dt=0.1, xstar=nothing) -> (ts, xs, rs, Vs)

Integrate the SPNN-QVI dynamics using OrdinaryDiffEq.jl (adaptive stepping).
Uses the same return signature as `solve_qvi_ode` for drop-in replacement.
Terminates early when the projection residual drops below `cfg.tol`.

The `cfg.dt` field is ignored (adaptive stepping chooses its own steps).
"""
function solve_qvi_diffeq(prob::QVIProblem, cfg::SolverConfig;
                           solver = Tsit5(),
                           save_dt::Float64 = 0.1,
                           xstar::Union{Nothing,Vector{Float64}} = nothing,
                           abstol::Float64 = 1e-8,
                           reltol::Float64 = 1e-6,
                           terminate_on_tol::Bool = true,
                           stop_fn = nothing,
                           return_sol::Bool = false,
                           compute_rs::Bool = true,
                           dtmax_override::Union{Nothing,Float64} = nothing)
    # RHS: dx/dt = λ(T(x) - x)
    function rhs!(dx, x, p, t)
        qvi, sc = p
        Tx = T_map(x, qvi, sc)
        @. dx = sc.lambda * (Tx - x)
        return nothing
    end

    # Stopping: either a caller-supplied rule (evaluated at accepted endpoints),
    # or the native residual rule ‖r(x)‖ < tol, or none.
    cb = nothing
    if stop_fn !== nothing
        cb = DiscreteCallback((u, t, integrator) -> stop_fn(u, t), terminate!)
    elseif terminate_on_tol
        function check_converged(u, t, integrator)
            qvi, sc = integrator.p
            _, rnorm = residual(u, qvi, sc)
            return rnorm < sc.tol
        end
        cb = DiscreteCallback(check_converged, terminate!)
    end

    # Solve
    ode_prob = ODEProblem(rhs!, copy(prob.x0), (0.0, cfg.T), (prob, cfg))
    sol = solve(ode_prob, solver;
        abstol   = abstol,
        reltol   = reltol,
        dtmax    = dtmax_override === nothing ? max(save_dt, 0.1) : dtmax_override,
        saveat   = save_dt,
        callback = cb,
        maxiters = cfg.maxiter,
    )

    # Convert to same format as solve_qvi_ode
    ts = collect(sol.t)
    xs = [copy(u) for u in sol.u]
    rs = compute_rs ? Float64[residual(u, prob, cfg)[2] for u in sol.u] : Float64[]
    Vs = Float64[xstar === nothing ? NaN : 0.5 * norm(u - xstar)^2 for u in sol.u]

    return_sol && return ts, xs, rs, Vs, sol
    return ts, xs, rs, Vs
end

"""
    solve_qvi_fixedpoint(prob, cfg; h=1/cfg.lambda, stop_fn=nothing, maxiter=cfg.maxiter)

Run the forward-Euler / Krasnoselskii--Mann iteration
  x_{k+1} = (1 - λh) x_k + λh T(x_k),
which at h = 1/λ is the fixed-point (Picard) iteration x_{k+1} = T(x_k)
(for M = I: the projection algorithm of Noor 1988, Algorithm 3.1 with g = id).

`stop_fn(x, k) -> Bool` is evaluated after every completed outer iterate; when
it returns true the iteration stops with `stopped = true`. Without `stop_fn`
the iteration runs to `maxiter` (no native stopping — callers supply the rule).

Returns a NamedTuple `(x_final, iterations, time_seconds, stopped)`.
"""
function solve_qvi_fixedpoint(prob::QVIProblem, cfg::SolverConfig;
                              h::Float64 = 1.0 / cfg.lambda,
                              stop_fn = nothing,
                              maxiter::Int = cfg.maxiter)
    θ = cfg.lambda * h
    @assert 0 < θ <= 1 "solve_qvi_fixedpoint requires λh ∈ (0, 1]"
    x = copy(prob.x0)
    t_start = time()
    k = 0
    stopped = false
    for iter in 1:maxiter
        k = iter
        Tx = T_map(x, prob, cfg)
        @. x = (1 - θ) * x + θ * Tx
        if stop_fn !== nothing && stop_fn(x, k)
            stopped = true
            break
        end
        if any(!isfinite, x) || norm(x) > 1e15
            break
        end
    end
    return (x_final = x, iterations = k, time_seconds = time() - t_start, stopped = stopped)
end

"""
    solve_qvi_ode(prob, cfg; save_dt=0.1) -> (ts, xs, rs, Vs)

Integrate the dynamics and return time-series data for plotting.
Returns vectors of (time, state, residual_norm, Lyapunov_value).
Requires xstar (the known solution) for Lyapunov computation.
"""
function solve_qvi_ode(prob::QVIProblem, cfg::SolverConfig;
                       save_dt::Float64=0.1,
                       xstar::Union{Nothing,Vector{Float64}}=nothing)
    x = copy(prob.x0)
    dt = cfg.dt
    λ = cfg.lambda
    T_final = cfg.T

    n_save = Int(ceil(T_final / save_dt)) + 1
    ts = Float64[0.0]
    xs = [copy(x)]
    _, rn = residual(x, prob, cfg)
    rs = Float64[rn]
    Vs = Float64[xstar === nothing ? NaN : 0.5 * norm(x - xstar)^2]

    t = 0.0
    t_last_save = 0.0

    while t < T_final
        Tx = T_map(x, prob, cfg)
        x = x + dt * λ * (Tx - x)
        t += dt

        if t - t_last_save >= save_dt - 1e-12
            _, rn = residual(x, prob, cfg)
            push!(ts, t)
            push!(xs, copy(x))
            push!(rs, rn)
            push!(Vs, xstar === nothing ? NaN : 0.5 * norm(x - xstar)^2)
            t_last_save = t
        end

        if any(isnan, x) || norm(x) > 1e15
            break
        end
    end

    return ts, xs, rs, Vs
end
