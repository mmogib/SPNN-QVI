module SPNNQVI

# === Imports ===
using LinearAlgebra, Printf, Random, Dates, Statistics, SparseArrays
using JuMP, HiGHS
using OrdinaryDiffEq, DiffEqCallbacks

# === Includes (dependency order) ===
include("types.jl")
include("projection.jl")
include("solver.jl")
include("utils.jl")
include("io_utils.jl")
include("problems.jl")

# === Exports ===
export
    # Types
    QVIProblem, SolverConfig, SolverResult, FactorInverse,
    # Projection
    metric_projection, metric_projection_translated,
    metric_project_box, metric_project_box_cholesky,
    # Solver
    T_map, residual, solve_qvi, solve_qvi_ode, solve_qvi_diffeq,
    solve_qvi_fixedpoint,
    # Utilities
    compute_residual, V_euclid, V_metric, time_to_tol, elapsed_str,
    reference_residual, reference_residual_lb, with_x0, norm_M,
    setup_logging, teardown_logging, TeeIO,
    # Problems
    get_problem, list_problems

end # module
