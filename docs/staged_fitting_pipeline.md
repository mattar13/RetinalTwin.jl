# Staged Retinal Twin Fitting Pipeline

This document describes the implemented fitting workflow in `src/fitting/staged_pipeline.jl`.

## What is implemented

- A structured ERG dataset type (`ERGDataSet`) with intensity panels and shared timebase.
- Preprocessing (`preprocess_erg`) with baseline subtraction and optional low-pass filtering.
- Feature extraction (`extract_erg_features`) for:
  - a-wave amplitude / implicit time
  - b-wave amplitude / implicit time
  - OP summary amplitude and dominant OP frequency
- Stage-based optimization (`fit_retinal_twin_staged`) with parameter freezing by stage:
  - **Stage A**: photoreceptor / a-wave window
  - **Stage B**: ON-bipolar / b-wave window
  - **Stage C**: amacrine / OP window
  - **Stage D**: Müller / slow window
- Two fitting modes:
  - `mode=:efficient` for fast iteration
  - `mode=:accurate` for denser search
- Parameter uncertainty estimation via local curvature (finite-difference Hessian diagonal approximation).
- Artifacts:
  - Fit summary plots (`plot_fit_diagnostics`)
  - Parameter estimate datasheet (`save_fit_datasheet`)

## Why staged fitting

The staged workflow avoids updating unrelated parameter groups when fitting a specific ERG component. This reduces search dimensionality and improves convergence consistency.

## Output products

`plot_fit_diagnostics` writes:

1. `fit_traces.png` — observed vs fitted traces across intensities.
2. `fit_residuals.png` — residual map (intensity × time).
3. `fit_parameters.png` — fitted parameter estimates with 95% confidence intervals.

`save_fit_datasheet` writes a CSV with:

- `stage`
- `block`
- `parameter`
- `estimate`
- `std_error`
- `ci95_lower`
- `ci95_upper`
- `certainty`
- `curvature`

## Running the demo

Use the synthetic-data demonstration:

```bash
julia --project=. examples/run_staged_fitting_pipeline.jl
```

This script writes all outputs under:

`examples/output/staged_fit/`
