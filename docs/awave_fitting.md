# A-Wave Parameter Fitting

Fits photoreceptor model parameters to real ERG data using Optim.jl.

## How it works

1. **Parameter discovery** — `fittable_params([:PHOTO])` reads `retinal_params.csv` and returns all non-fixed photoreceptor parameters with their bounds (~45 params after excluding reversal potentials, geometric constants, etc.)

2. **Two-phase optimization**:
   - **NelderMead** (500 iterations) — derivative-free, robust to ODE solver failures from extreme parameter combinations
   - **LBFGS** (100 iterations) — gradient-based refinement using finite-difference gradients for precision near the optimum

3. **Loss function** — Mean squared error between simulated and real ERG traces within a specified time window, summed across all stimulus intensities. Simulated traces are interpolated onto the real data's time grid.

4. **Uncertainty** — Finite-difference Hessian diagonal at the optimum provides per-parameter standard errors and 95% confidence intervals. Off-diagonal Hessian elements detect correlated parameter pairs (flagged when |r| > 0.7).

## Running

```bash
julia --project=. examples/run_awave_optim_fitting.jl
```

Requires real BaCl+LAP4 ERG data loaded via ElectroPhysiology.jl. Edit the `erg_dir` path in the script to point to your data.

## Outputs

Written to `examples/output/awave_optim_fit/`:

| File | Description |
|------|-------------|
| `fit_traces.png` | Real vs fitted ERG traces per intensity |
| `fit_residuals.png` | Residual traces within the fit window |
| `fit_parameters.png` | Parameter estimates with 95% CI error bars |
| `fit_correlations.png` | Correlated parameter pairs (if any detected) |
| `awave_fit_params.csv` | Full parameter datasheet |

The CSV datasheet contains:
- `cell_type`, `parameter` — which parameter
- `estimate` — fitted value
- `std_error` — standard error from Hessian
- `ci95_lower`, `ci95_upper` — 95% confidence interval
- `certainty` — confidence score (0–1, higher = more certain)
- `correlated_with` — other parameters this one correlates with

## Extending to b-wave

The same `fit_erg` function works for any ERG component — just change `cell_types` and `time_window`:

```julia
# B-wave fitting (ON-bipolar parameters)
result_b = fit_erg(model, u0, params;
    cell_types = [:ONBC],
    stimuli = stimuli,
    real_t = real_t,
    real_traces = real_traces,
    time_window = (0.02, 0.15),
)

# Multi-stage: fit a-wave first, then b-wave with frozen photoreceptor params
result_a = fit_erg(model, u0, params; cell_types=[:PHOTO], ...)
result_b = fit_erg(model, u0, result_a.params; cell_types=[:ONBC], ...)
```

## Fixed vs fittable parameters

Parameters in `retinal_params.csv` with `Fixed=true` are never optimized. These include:
- Reversal/equilibrium potentials (set by ionic concentrations)
- Membrane capacitance (measured property)
- Geometric constants (cell dimensions)
- Applied current (experimental protocol)

To add or remove parameters from fitting, edit the `Fixed` column in the CSV.
