# SPNN-QVI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19687760.svg)](https://doi.org/10.5281/zenodo.19687760)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Julia implementation accompanying the paper

> **Scaled Projection Neural Network for Quasi-Variational Inequalities**
> M. Alshahrani and Q. H. Ansari (2026), submitted to *Neural Networks*.

**Software archive:** https://doi.org/10.5281/zenodo.19687760

The solver integrates the continuous-time scaled projection neural network

$$\frac{dx}{dt} = \lambda \bigl[P_{\mathfrak{S}(x),M}\bigl(x - \alpha\,M\,F(x)\bigr) - x\bigr],$$

where $\mathfrak{S}(x) = m(x) + \mathcal{S}$ is a state-dependent constraint set and $M$ is a fixed symmetric positive-definite metric matrix. Adaptive ODE integration is performed via `OrdinaryDiffEq.jl` (Tsit5 by default, AutoTsit5/Rosenbrock23 for stiff problems).

## Repository layout

```
src/           Julia module SPNNQVI
  SPNNQVI.jl   Entry module (includes + exports)
  types.jl     QVIProblem, SolverConfig, SolverResult
  solver.jl    Core dynamics: solve_qvi (Euler), solve_qvi_diffeq (adaptive)
  projection.jl Metric projection via coordinate descent
  problems.jl  9 QVI problem definitions
  utils.jl     Residual, Lyapunov, time_to_tol
  io_utils.jl  TeeIO logging, CSV export
test/          Test suite
scripts/       Experiment scripts (s10 .. s70)
results/       Output directory (contents git-ignored)
```

## Requirements

- Julia 1.10 or later
- Dependencies are pinned in `Manifest.toml` for exact reproducibility.
  Experienced users may regenerate them from `Project.toml` if they prefer.

## Reproducing the experiments

```bash
git clone https://github.com/mmogib/SPNN-QVI.git
cd SPNN-QVI
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# smoke test (basic verification)
julia --project=. scripts/s10_smoke_test.jl

# run the full experiment suite (writes to results/)
julia --project=. scripts/s20_example1.jl
julia --project=. scripts/s30_example2.jl
julia --project=. scripts/s35_contraction_sweep.jl
julia --project=. scripts/s36_stepsize_sweep.jl
julia --project=. scripts/s40_random_benchmark.jl
julia --project=. scripts/s55_beyond_theory.jl
julia --project=. scripts/s60_obstacle.jl
julia --project=. scripts/s62_gnep.jl
julia --project=. scripts/s64_nonlinear.jl
julia --project=. scripts/s66_scaling.jl
julia --project=. scripts/s68_noor_comparison.jl

# generate paper figures (7 PDFs via PGFPlotsX)
julia --project=. scripts/s70_figures.jl

# run tests
julia --project=. test/runtests.jl
```

Each script writes a CSV and a log into the matching subfolder under `results/`.

## Problems implemented

| ID | Name                  | n     | Source                          |
|----|-----------------------|-------|----------------------------------|
| 1  | 2D affine (rotated Q) | 2     | Model problem                    |
| 2  | 2D rotated box        | 2     | Model problem                    |
| 3  | 2D nonlinear          | 2     | Model problem                    |
| 5  | Implicit obstacle     | 20    | Noor 1988; Baiocchi–Capelo 1984  |
| 6  | GNEP Cournot duopoly  | 2     | Harker 1991; Facchinei–Kanzow 2010 |
| 7  | Nonlinear monotone    | 5     | Xia–Wang 2004; Hu–Wang 2006      |
| 8  | Random high-dim       | 10–50 | Solodov–Tseng 1996               |
| 9  | Noor 2003 Example 4.1 | 2     | Noor 2003                        |

## Citation

If you use this code, please cite both the software and the paper.

**Software (Zenodo):**

```bibtex
@software{AlshahraniAnsari_SPNNQVI_2026,
  author  = {Alshahrani, Mohammed and Ansari, Qamrul Hasan},
  title   = {{SPNN-QVI: Scaled Projection Neural Network for
             Quasi-Variational Inequalities}},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.19687760},
  url     = {https://doi.org/10.5281/zenodo.19687760}
}
```

**Paper:** a BibTeX entry will be added here once the paper has a DOI.

## License

MIT — see `LICENSE`.
