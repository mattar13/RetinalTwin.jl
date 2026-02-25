# Retinal Digital Twin — Parameter Fitting & Bayesian Inference Specification

## For Implementation by Claude Code / Codex

**Project context:** This document specifies the parameter estimation pipeline for a retinal digital twin that simulates ERG (electroretinogram) waveforms using a 193-ODE Morris-Lecar-based neural population model implemented in Julia. The model spec is in `retinal_digital_twin_spec.md`. The user (Matt Tarchick) is a computational neuroscientist with deep ERG expertise. He has intensity-response series data (multiple flash intensities, one ERG trace per intensity) and wants both point estimates and full posterior distributions over model parameters.

---


> **Implementation note (current repository):** A practical staged fitter based on this specification is implemented in `src/fitting/staged_pipeline.jl`, with usage notes in `docs/staged_fitting_pipeline.md` and a runnable demo in `examples/run_staged_fitting_pipeline.jl`.

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Parameter Space Definition](#2-parameter-space-definition)
3. [Data Interface](#3-data-interface)
4. [Loss Function Design](#4-loss-function-design)
5. [Stage 1: Point Estimation via Gradient-Based Optimization](#5-stage-1-point-estimation)
6. [Stage 2: Bayesian Inference via Turing.jl + NUTS](#6-stage-2-bayesian-inference)
7. [Staged Fitting Pipeline](#7-staged-fitting-pipeline)
8. [Interactive Mode](#8-interactive-mode)
9. [Diagnostics & Visualization](#9-diagnostics--visualization)
10. [Code Architecture](#10-code-architecture)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Dependencies](#12-dependencies)

---

## 1. Strategy Overview

### 1.1 The Problem

We have a 193-ODE dynamical system with ~60–80 tunable biophysical parameters. We observe only the ERG field potential — a single scalar time series that is a weighted sum of all cell population currents. We have this observation at multiple light intensities. We want to:

1. Find the parameter values that best reproduce the observed ERG waveforms across all intensities simultaneously
2. Quantify uncertainty on those parameters (posterior distributions)
3. Do this in a way that is robust to local minima and leverages domain knowledge

### 1.2 Why This Is Hard (and How We Handle It)

**Challenge 1: High-dimensional parameter space.**
We have ~70 parameters. Naively sampling a 70D posterior with MCMC would require millions of samples. **Solution:** Staged fitting — decompose the problem into 5 sub-problems of ~8–15 parameters each, exploiting the known mapping between ERG components and cell types. Then do a final global refinement.

**Challenge 2: The ERG is a single observable.**
We can't directly observe individual cell voltages — we see only the summed field potential. **Solution:** The intensity-response series provides strong constraints. Different intensities activate different pathways (scotopic vs. photopic) and produce features with different sensitivities to different parameters. Additionally, ERG component features (a-wave timing, b-wave amplitude, OP frequency) each constrain different parameter subsets.

**Challenge 3: ODE parameter estimation is prone to local minima.**
The loss landscape is non-convex. **Solution:** Three-layer approach:
1. Global search with multi-start optimization (BlackBoxOptim or TikTak)
2. Gradient-based refinement with ADAM → L-BFGS (using adjoint sensitivity for gradients)
3. Bayesian posterior sampling with NUTS (initialized at the MAP estimate)

**Challenge 4: Computational cost of each forward solve.**
Each ODE solve takes ~0.1–1 second. MCMC needs thousands of solves. **Solution:** Use adjoint sensitivity methods (SciMLSensitivity.jl) for O(1)-in-parameters gradient computation. Use `InterpolatingAdjoint` with checkpointing for memory efficiency. For MCMC, use NUTS which is efficient in the number of likelihood evaluations needed per effective sample.

### 1.3 Algorithm Selection Rationale

| Stage | Method | Package | Why This One |
|-------|--------|---------|-------------|
| Global exploration | Multi-start + BBO | OptimizationBBO, MultistartOptimization | Escape local minima without gradients |
| Gradient refinement | ADAM → L-BFGS | OptimizationOptimisers, OptimizationOptimJL | ADAM handles noisy/flat landscapes; L-BFGS gives precise convergence |
| Gradients through ODE | Adjoint sensitivity | SciMLSensitivity.jl | O(1) in number of parameters; memory-efficient with checkpointing |
| Posterior sampling | NUTS (No-U-Turn Sampler) | Turing.jl | Gold standard for continuous posteriors; adapts step size automatically; efficient for correlated parameters |
| Posterior diagnostics | R-hat, ESS, trace plots | MCMCChains.jl | Standard MCMC convergence diagnostics |

### 1.4 Pipeline Summary

```
┌─────────────────────────────────────────────────────┐
│              STAGED POINT ESTIMATION                 │
│                                                      │
│  Stage A: Photoreceptor params → a-wave              │
│      ↓ (freeze)                                      │
│  Stage B: ON-bipolar/mGluR6 params → b-wave          │
│      ↓ (freeze)                                      │
│  Stage C: Amacrine params → OPs                      │
│      ↓ (freeze)                                      │
│  Stage D: Müller/RPE params → P3 + c-wave            │
│      ↓ (freeze)                                      │
│  Stage E: ERG weights → overall scaling              │
│      ↓                                               │
│  Stage F: Global refinement (all params unfrozen)    │
└─────────────┬───────────────────────────────────────┘
              │ MAP estimate (best-fit parameters)
              ▼
┌─────────────────────────────────────────────────────┐
│           BAYESIAN POSTERIOR SAMPLING                 │
│                                                      │
│  Initialize NUTS chains at MAP estimate              │
│  Run 4 chains × 2000 samples (after warmup)          │
│  Compute R-hat, ESS, posterior intervals             │
│  Generate posterior predictive checks                │
└─────────────────────────────────────────────────────┘
```

---

## 2. Parameter Space Definition

### 2.1 Parameter Groups

Parameters are organized into groups that correspond to fitting stages. Each parameter has: a name, a default value, biophysically motivated bounds, a prior distribution for Bayesian inference, and a flag indicating whether it is currently free or frozen.

#### Group A: Phototransduction (Constrained by a-wave)

These parameters control the photoreceptor response and thus the a-wave.

| Parameter | Symbol | Default | Lower | Upper | Prior | Units |
|-----------|--------|---------|-------|-------|-------|-------|
| Rod R* inactivation τ | `τ_R_rod` | 80.0 | 30.0 | 200.0 | LogNormal(log(80), 0.3) | ms |
| Cone R* inactivation τ | `τ_R_cone` | 10.0 | 3.0 | 30.0 | LogNormal(log(10), 0.3) | ms |
| Rod PDE gain | `γ_rod` | 5.0 | 1.0 | 20.0 | LogNormal(log(5), 0.5) | — |
| Cone PDE gain | `γ_cone` | 3.0 | 0.5 | 15.0 | LogNormal(log(3), 0.5) | — |
| Rod CNG max conductance | `g_CNG_rod` | 20.0 | 5.0 | 50.0 | LogNormal(log(20), 0.3) | nS |
| Cone CNG max conductance | `g_CNG_cone` | 30.0 | 10.0 | 60.0 | LogNormal(log(30), 0.3) | nS |
| Ca feedback cooperativity | `n_Ca` | 4.0 | 1.0 | 6.0 | Normal(4, 1) | — |
| I_H conductance (rod) | `g_H_rod` | 2.0 | 0.5 | 8.0 | LogNormal(log(2), 0.5) | nS |
| I_H time constant | `τ_H` | 50.0 | 20.0 | 100.0 | LogNormal(log(50), 0.3) | ms |
| Quantum efficiency (rod) | `η_rod` | 0.67 | 0.3 | 1.0 | Beta(6.7, 3.3) | — |
| cGMP synthesis rate | `α_G` | 20.0 | 5.0 | 50.0 | LogNormal(log(20), 0.4) | µM/ms |

**Total: 11 parameters**

#### Group B: ON-Pathway / b-wave

| Parameter | Symbol | Default | Lower | Upper | Prior | Units |
|-----------|--------|---------|-------|-------|-------|-------|
| mGluR6 time constant | `τ_mGluR6` | 30.0 | 10.0 | 80.0 | LogNormal(log(30), 0.3) | ms |
| TRPM1 max conductance | `g_TRPM1` | 10.0 | 2.0 | 30.0 | LogNormal(log(10), 0.4) | nS |
| ON-BC g_Ca | `g_Ca_on` | 4.0 | 1.0 | 12.0 | LogNormal(log(4), 0.4) | nS |
| ON-BC g_K | `g_K_on` | 8.0 | 2.0 | 20.0 | LogNormal(log(8), 0.3) | nS |
| ON-BC phi | `φ_on` | 0.067 | 0.01 | 0.2 | LogNormal(log(0.067), 0.4) | — |
| ON-BC V3 | `V3_on` | 12.0 | -10.0 | 30.0 | Normal(12, 8) | mV |
| mGluR6 alpha scaling | `α_mGluR6` | 1.0 | 0.2 | 3.0 | LogNormal(0, 0.4) | — |
| Glu release τ (ON-BC) | `τ_Glu_on` | 5.0 | 1.0 | 20.0 | LogNormal(log(5), 0.4) | ms |

**Total: 8 parameters**

#### Group C: Oscillatory Potentials (Amacrine Network)

| Parameter | Symbol | Default | Lower | Upper | Prior | Units |
|-----------|--------|---------|-------|-------|-------|-------|
| A2 g_Ca | `g_Ca_a2` | 8.0 | 2.0 | 20.0 | LogNormal(log(8), 0.3) | nS |
| A2 g_K | `g_K_a2` | 12.0 | 4.0 | 25.0 | LogNormal(log(12), 0.3) | nS |
| A2 phi | `φ_a2` | 0.2 | 0.05 | 0.5 | LogNormal(log(0.2), 0.3) | — |
| A2 V3 | `V3_a2` | -10.0 | -25.0 | 5.0 | Normal(-10, 6) | mV |
| GABA-AC g_Ca | `g_Ca_gaba` | 8.0 | 2.0 | 20.0 | LogNormal(log(8), 0.3) | nS |
| GABA-AC phi | `φ_gaba` | 0.15 | 0.03 | 0.4 | LogNormal(log(0.15), 0.3) | — |
| Glycine release τ | `τ_Gly` | 4.0 | 1.0 | 10.0 | LogNormal(log(4), 0.3) | ms |
| GABA release τ | `τ_GABA` | 7.0 | 2.0 | 15.0 | LogNormal(log(7), 0.3) | ms |
| A2↔GABA coupling (Gly→GABA) | `g_Gly_gaba` | 10.0 | 2.0 | 25.0 | LogNormal(log(10), 0.3) | nS |
| A2↔GABA coupling (GABA→A2) | `g_GABA_a2` | 10.0 | 2.0 | 25.0 | LogNormal(log(10), 0.3) | nS |
| ON-BC→A2 coupling | `g_Glu_a2` | 8.0 | 2.0 | 20.0 | LogNormal(log(8), 0.3) | nS |

**Total: 11 parameters**

#### Group D: Slow Components (Müller + RPE)

| Parameter | Symbol | Default | Lower | Upper | Prior | Units |
|-----------|--------|---------|-------|-------|-------|-------|
| Müller Kir endfoot | `g_Kir_end` | 5.0 | 1.0 | 15.0 | LogNormal(log(5), 0.4) | nS |
| Müller Kir stalk | `g_Kir_stalk` | 2.0 | 0.5 | 8.0 | LogNormal(log(2), 0.4) | nS |
| K+ diffusion τ | `τ_K_diff` | 200.0 | 50.0 | 1000.0 | LogNormal(log(200), 0.5) | ms |
| K+ release coupling | `α_K` | 0.001 | 0.0001 | 0.01 | LogNormal(log(0.001), 0.5) | — |
| RPE time constant | `τ_RPE` | 3000.0 | 1000.0 | 8000.0 | LogNormal(log(3000), 0.3) | ms |
| RPE K apical g | `g_K_apical` | 5.0 | 1.0 | 15.0 | LogNormal(log(5), 0.4) | nS |
| RPE K+ flux scaling | `α_K_RPE` | 0.001 | 0.0001 | 0.01 | LogNormal(log(0.001), 0.5) | — |

**Total: 7 parameters**

#### Group E: ERG Weights

| Parameter | Symbol | Default | Lower | Upper | Prior | Units |
|-----------|--------|---------|-------|-------|-------|-------|
| Rod weight | `w_rod` | 1.0 | 0.1 | 5.0 | LogNormal(0, 0.5) | — |
| Cone weight | `w_cone` | 0.5 | 0.05 | 3.0 | LogNormal(log(0.5), 0.5) | — |
| ON-BC weight | `w_on` | -2.0 | -8.0 | -0.5 | Normal(-2, 1.5) | — |
| OFF-BC weight | `w_off` | -1.0 | -5.0 | -0.1 | Normal(-1, 1) | — |
| A2 weight | `w_a2` | 0.3 | 0.01 | 2.0 | LogNormal(log(0.3), 0.5) | — |
| GABA weight | `w_gaba` | 0.3 | 0.01 | 2.0 | LogNormal(log(0.3), 0.5) | — |
| Müller weight | `w_muller` | -1.5 | -5.0 | -0.1 | Normal(-1.5, 1) | — |
| RPE weight | `w_rpe` | -1.0 | -5.0 | -0.1 | Normal(-1, 1) | — |

**Total: 8 parameters**

#### Group F: Sensitivity / Scaling

| Parameter | Symbol | Default | Lower | Upper | Prior | Units |
|-----------|--------|---------|-------|-------|-------|-------|
| Rod sensitivity scaling | `I_ref_rod` | 1.0 | 0.1 | 10.0 | LogNormal(0, 0.5) | — |
| Cone sensitivity scaling | `I_ref_cone` | 100.0 | 10.0 | 1000.0 | LogNormal(log(100), 0.5) | — |
| Noise σ (observation) | `σ_obs` | 10.0 | 1.0 | 100.0 | InverseGamma(2, 30) | µV |

**Total: 3 parameters**

### 2.2 Total Parameter Count

| Group | Count | Fitting Stage |
|-------|-------|---------------|
| A: Phototransduction | 11 | a-wave |
| B: ON-Pathway | 8 | b-wave |
| C: Amacrine / OPs | 11 | OPs |
| D: Slow Components | 7 | P3 + c-wave |
| E: ERG Weights | 8 | Scaling |
| F: Sensitivity / Noise | 3 | Global |
| **Total** | **48** | |

### 2.3 Parameter Vector Structure

```julia
using ComponentArrays

function build_parameter_vector(; kwargs...)
    return ComponentArray(
        # Group A: Phototransduction
        phototrans = ComponentArray(
            τ_R_rod = 80.0, τ_R_cone = 10.0,
            γ_rod = 5.0, γ_cone = 3.0,
            g_CNG_rod = 20.0, g_CNG_cone = 30.0,
            n_Ca = 4.0,
            g_H_rod = 2.0, τ_H = 50.0,
            η_rod = 0.67, α_G = 20.0
        ),
        # Group B: ON-Pathway
        on_pathway = ComponentArray(
            τ_mGluR6 = 30.0, g_TRPM1 = 10.0,
            g_Ca_on = 4.0, g_K_on = 8.0,
            φ_on = 0.067, V3_on = 12.0,
            α_mGluR6 = 1.0, τ_Glu_on = 5.0
        ),
        # Group C: Amacrine / OPs
        amacrine = ComponentArray(
            g_Ca_a2 = 8.0, g_K_a2 = 12.0,
            φ_a2 = 0.2, V3_a2 = -10.0,
            g_Ca_gaba = 8.0, φ_gaba = 0.15,
            τ_Gly = 4.0, τ_GABA = 7.0,
            g_Gly_gaba = 10.0, g_GABA_a2 = 10.0,
            g_Glu_a2 = 8.0
        ),
        # Group D: Slow components
        slow = ComponentArray(
            g_Kir_end = 5.0, g_Kir_stalk = 2.0,
            τ_K_diff = 200.0, α_K = 0.001,
            τ_RPE = 3000.0, g_K_apical = 5.0,
            α_K_RPE = 0.001
        ),
        # Group E: ERG Weights
        erg_weights = ComponentArray(
            w_rod = 1.0, w_cone = 0.5,
            w_on = -2.0, w_off = -1.0,
            w_a2 = 0.3, w_gaba = 0.3,
            w_muller = -1.5, w_rpe = -1.0
        ),
        # Group F: Sensitivity / Noise
        scaling = ComponentArray(
            I_ref_rod = 1.0, I_ref_cone = 100.0,
            σ_obs = 10.0
        )
    )
end
```

### 2.4 Parameter Transformations

Many parameters are positive-definite (conductances, time constants). For gradient-based optimization and MCMC, work in **transformed space** where parameters are unconstrained:

```julia
"""
Transform a bounded parameter to unconstrained space.
- Positive parameters: log transform
- Bounded parameters: logit transform
- Unconstrained (e.g., V3): identity
"""
function to_unconstrained(θ_raw, bounds)
    θ_unc = similar(θ_raw)
    for i in eachindex(θ_raw)
        lb, ub = bounds[i]
        if lb == -Inf && ub == Inf
            θ_unc[i] = θ_raw[i]
        elseif lb == 0.0 && ub == Inf
            θ_unc[i] = log(θ_raw[i])
        else
            # Logit transform for bounded parameters
            x_scaled = (θ_raw[i] - lb) / (ub - lb)
            θ_unc[i] = log(x_scaled / (1.0 - x_scaled))
        end
    end
    return θ_unc
end

function from_unconstrained(θ_unc, bounds)
    θ_raw = similar(θ_unc)
    for i in eachindex(θ_unc)
        lb, ub = bounds[i]
        if lb == -Inf && ub == Inf
            θ_raw[i] = θ_unc[i]
        elseif lb == 0.0 && ub == Inf
            θ_raw[i] = exp(θ_unc[i])
        else
            θ_raw[i] = lb + (ub - lb) * logistic(θ_unc[i])
        end
    end
    return θ_raw
end
```

Turing.jl handles transforms automatically with `bijector()` when using constrained distributions, so the Bayesian stage does not need manual transforms. The point estimation stage uses Optimization.jl's built-in `lb`/`ub` support.

---

## 3. Data Interface

### 3.1 Expected Data Format

```julia
"""
    ERGDataSet

A collection of ERG traces at different flash intensities.
"""
struct ERGDataSet
    # Vector of flash intensities (log units or absolute)
    intensities::Vector{Float64}
    
    # Time vector (shared across all traces, or per-trace)
    # Assumed uniform sampling; typical: 0.1 ms or 1 ms spacing
    t::Vector{Float64}  
    
    # ERG traces: matrix of size (n_timepoints × n_intensities)
    # Each column is one ERG trace in µV
    traces::Matrix{Float64}
    
    # Optional: standard deviation per point (for weighted fitting)
    # Same shape as traces, or scalar
    σ::Union{Float64, Matrix{Float64}}
    
    # Metadata
    species::Symbol          # :human, :mouse, :macaque
    adaptation::Symbol       # :dark_adapted, :light_adapted
    flash_duration::Float64  # ms
    pre_flash_time::Float64  # ms of baseline before flash
end
```

### 3.2 Data Loading

```julia
"""
    load_erg_data(filepath; format=:csv)

Load ERG data from file. Supports CSV and MAT formats.

Expected CSV format:
- First column: time (ms)
- Subsequent columns: ERG traces at each intensity
- First row header: "time,intensity_1,intensity_2,..."
  where intensity values are in the header

Returns: ERGDataSet
"""
function load_erg_data(filepath; format=:csv, 
                       flash_duration=10.0,
                       pre_flash_time=200.0,
                       species=:human,
                       adaptation=:dark_adapted)
    # Implementation reads file, extracts time vector,
    # parses intensity values from headers,
    # and constructs ERGDataSet
end
```

### 3.3 Data Preprocessing

Before fitting, preprocess the data:

```julia
"""
    preprocess_erg(data::ERGDataSet)

1. Baseline subtraction: subtract mean of pre-flash period
2. Optional: low-pass filter to remove high-frequency noise
3. Optional: extract OP-filtered traces (bandpass 75-300 Hz)
4. Compute feature targets (a-wave amp, b-wave amp, implicit times)
"""
function preprocess_erg(data::ERGDataSet)
    # Baseline subtract
    baseline_mask = data.t .< 0.0  # pre-flash
    for i in axes(data.traces, 2)
        bl = mean(data.traces[baseline_mask, i])
        data.traces[:, i] .-= bl
    end
    
    # Extract features for feature-based loss components
    features = extract_erg_features(data)
    
    return data, features
end
```

### 3.4 Feature Extraction

```julia
"""
    ERGFeatures

Extracted scalar features from an ERG trace for feature-based loss.
"""
struct ERGFeatures
    # Per-intensity features (vectors, one element per intensity)
    a_wave_amplitude::Vector{Float64}     # µV (negative)
    a_wave_implicit_time::Vector{Float64} # ms (time to a-wave trough)
    b_wave_amplitude::Vector{Float64}     # µV (positive, measured from a-wave trough)
    b_wave_implicit_time::Vector{Float64} # ms (time to b-wave peak)
    op_sum_amplitude::Vector{Float64}     # µV (sum of OP amplitudes)
    op_dominant_freq::Vector{Float64}     # Hz (peak frequency of OP band)
    # Naka-Rushton fits
    b_wave_Vmax::Float64                  # µV
    b_wave_K::Float64                     # half-saturation intensity
    b_wave_n::Float64                     # Hill coefficient
end

"""
    extract_erg_features(data::ERGDataSet)

Automated extraction of canonical ERG features from each trace.
"""
function extract_erg_features(data::ERGDataSet)
    n_intensities = length(data.intensities)
    
    a_amp = zeros(n_intensities)
    a_time = zeros(n_intensities)
    b_amp = zeros(n_intensities)
    b_time = zeros(n_intensities)
    op_sum = zeros(n_intensities)
    op_freq = zeros(n_intensities)
    
    flash_idx = findfirst(data.t .>= 0.0)
    
    for i in 1:n_intensities
        trace = data.traces[:, i]
        
        # a-wave: minimum in first 30 ms after flash
        a_window = flash_idx:(flash_idx + round(Int, 30.0 / step(data.t)))
        a_amp[i], a_rel_idx = findmin(trace[a_window])
        a_time[i] = data.t[a_window[a_rel_idx]]
        
        # b-wave: maximum in 30-120 ms after flash
        b_start = flash_idx + round(Int, 20.0 / step(data.t))
        b_end = flash_idx + round(Int, 120.0 / step(data.t))
        b_window = b_start:min(b_end, length(trace))
        b_peak, b_rel_idx = findmax(trace[b_window])
        b_amp[i] = b_peak - a_amp[i]  # measured from a-wave trough
        b_time[i] = data.t[b_window[b_rel_idx]]
        
        # OPs: bandpass 75-300 Hz, measure in 15-50 ms window
        # (requires DSP.jl)
        op_sum[i], op_freq[i] = extract_ops(trace, data.t)
    end
    
    # Fit Naka-Rushton to b-wave amplitude vs intensity
    Vmax, K, n = fit_naka_rushton(data.intensities, b_amp)
    
    return ERGFeatures(a_amp, a_time, b_amp, b_time, 
                       op_sum, op_freq, Vmax, K, n)
end
```

---

## 4. Loss Function Design

### 4.1 Multi-Component Loss

The loss function combines **waveform matching** (time-domain) and **feature matching** (scalar features). This hybrid approach is critical: pure waveform matching can chase noise, while pure feature matching loses temporal structure.

$$
\mathcal{L}(\theta) = \lambda_{wave} \cdot \mathcal{L}_{wave}(\theta) + \lambda_{feat} \cdot \mathcal{L}_{feat}(\theta) + \lambda_{reg} \cdot \mathcal{L}_{reg}(\theta)
$$

### 4.2 Waveform Loss

Weighted MSE between simulated and observed ERG traces across all intensities:

$$
\mathcal{L}_{wave}(\theta) = \sum_{k=1}^{N_{int}} \omega_k \sum_{j=1}^{N_t} W_j \left( V_{ERG}^{sim}(t_j; I_k, \theta) - V_{ERG}^{obs}(t_j; I_k) \right)^2
$$

where:
- $N_{int}$: number of intensities
- $\omega_k$: per-intensity weight (higher for intensities that show features of interest)
- $W_j$: temporal weighting function (see below)

**Temporal weighting $W_j$:** Different time windows carry different information. Weight them accordingly:

```julia
function temporal_weights(t; flash_onset=0.0)
    W = ones(length(t))
    for (i, ti) in enumerate(t)
        dt = ti - flash_onset
        if dt < 0
            W[i] = 0.1          # baseline: low weight
        elseif dt < 50
            W[i] = 3.0          # a-wave region: high weight
        elseif dt < 150
            W[i] = 5.0          # b-wave + OP region: highest weight
        elseif dt < 500
            W[i] = 1.0          # P3 region: moderate
        elseif dt < 5000
            W[i] = 0.5          # c-wave region: moderate-low
        else
            W[i] = 0.1          # late baseline
        end
    end
    return W ./ sum(W)  # normalize
end
```

**Per-intensity weights $\omega_k$:** Equal by default, but increase weight for intensities near rod saturation and in the mesopic range where both pathways contribute:

```julia
function intensity_weights(intensities)
    # Higher weight for intermediate intensities (most informative)
    # Lower weight for dimmest (noisy) and brightest (saturated)
    n = length(intensities)
    ω = ones(n)
    # Can be adjusted based on data quality
    return ω ./ sum(ω)
end
```

### 4.3 Feature Loss

Penalizes deviations in extracted scalar features. This provides a "coarse" signal that helps avoid local minima:

$$
\mathcal{L}_{feat}(\theta) = \sum_f \alpha_f \left(\frac{F_f^{sim}(\theta) - F_f^{obs}}{\sigma_f}\right)^2
$$

where $F_f$ are features (a-wave amplitude, b-wave timing, OP frequency, etc.) and $\sigma_f$ is a normalization scale for each feature.

```julia
function feature_loss(θ, data_features::ERGFeatures, 
                      sim_features::ERGFeatures)
    loss = 0.0
    n = length(data_features.a_wave_amplitude)
    
    # a-wave amplitude (normalize by max observed a-wave)
    σ_a = maximum(abs.(data_features.a_wave_amplitude)) + 1.0
    loss += sum(((sim_features.a_wave_amplitude .- 
                  data_features.a_wave_amplitude) ./ σ_a).^2) / n
    
    # a-wave timing (normalize by ~10 ms)
    loss += sum(((sim_features.a_wave_implicit_time .- 
                  data_features.a_wave_implicit_time) ./ 10.0).^2) / n
    
    # b-wave amplitude
    σ_b = maximum(data_features.b_wave_amplitude) + 1.0
    loss += 2.0 * sum(((sim_features.b_wave_amplitude .- 
                        data_features.b_wave_amplitude) ./ σ_b).^2) / n
    
    # b-wave timing
    loss += sum(((sim_features.b_wave_implicit_time .- 
                  data_features.b_wave_implicit_time) ./ 15.0).^2) / n
    
    # OP frequency (target 100-160 Hz, normalize by 30 Hz)
    loss += 0.5 * sum(((sim_features.op_dominant_freq .- 
                        data_features.op_dominant_freq) ./ 30.0).^2) / n
    
    # Naka-Rushton parameters
    loss += 2.0 * ((sim_features.b_wave_Vmax - data_features.b_wave_Vmax) / 
                    data_features.b_wave_Vmax)^2
    loss += ((sim_features.b_wave_K - data_features.b_wave_K) / 
              data_features.b_wave_K)^2
    
    return loss
end
```

### 4.4 Regularization Loss

Light regularization to keep parameters near biophysically plausible values. This is the log-prior in the Bayesian formulation:

$$
\mathcal{L}_{reg}(\theta) = -\sum_i \log p(\theta_i)
$$

In point estimation this acts as L2-like regularization toward the prior mean. In Bayesian inference, it becomes the actual prior.

### 4.5 Stage-Specific Loss Functions

During staged fitting, the loss is restricted to the relevant ERG features:

| Stage | Loss Components | Time Window |
|-------|----------------|-------------|
| A: Photoreceptor | a-wave amplitude + timing across intensities | 0–40 ms |
| B: ON-pathway | b-wave amplitude + timing; intensity-response curve | 20–150 ms |
| C: OPs | OP-filtered waveform (bandpass 75–300 Hz) | 15–60 ms |
| D: Slow | P3 (200–1000 ms) + c-wave (1000–5000 ms) | 200–5000 ms |
| E: ERG weights | Full waveform, scaling only | Full |
| F: Global | Full waveform + all features | Full |

```julia
"""
    stage_loss(stage::Symbol, θ, sim_result, data, features)

Compute stage-specific loss function.
"""
function stage_loss(stage::Symbol, θ, sim_result, data, features)
    if stage == :photoreceptor
        # Only penalize a-wave region across all intensities
        return awave_loss(θ, sim_result, data, features)
    elseif stage == :on_pathway
        return bwave_loss(θ, sim_result, data, features)
    elseif stage == :oscillatory
        return op_loss(θ, sim_result, data, features)
    elseif stage == :slow_components
        return slow_loss(θ, sim_result, data, features)
    elseif stage == :erg_weights
        return weight_loss(θ, sim_result, data, features)
    elseif stage == :global
        return full_loss(θ, sim_result, data, features)
    end
end
```

### 4.6 Handling Multiple Intensities

The forward model must simulate the retinal column at **each** flash intensity and combine the results:

```julia
"""
    simulate_intensity_series(θ, intensities; t_total=5000.0, dt=0.1)

Run the retinal column simulation at each intensity.
Returns matrix of ERG traces (n_timepoints × n_intensities).
"""
function simulate_intensity_series(θ, intensities; 
                                    t_total=5000.0, dt=0.1)
    n_int = length(intensities)
    col = build_column_from_params(θ)
    sidx = StateIndex(col)
    
    # Preallocate output
    t_vec = 0.0:dt:t_total
    erg_matrix = zeros(length(t_vec), n_int)
    
    for (k, I_k) in enumerate(intensities)
        # Update stimulus intensity
        col_k = @set col.stimulus.I_0 = I_k
        
        # Solve ODE
        u0 = dark_adapted_state(col_k, sidx)
        prob = ODEProblem(retinal_column_rhs!, u0, (0.0, t_total), (col_k, sidx))
        sol = solve(prob, Tsit5(); saveat=dt, 
                    abstol=1e-8, reltol=1e-6,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))
        
        # Compute ERG for this intensity
        erg_matrix[:, k] = compute_erg_trace(sol, col_k, sidx)
    end
    
    return t_vec, erg_matrix
end
```

**Performance note:** The intensity series can be parallelized across intensities using `Threads.@threads` or `EnsembleProblem` from DifferentialEquations.jl, since each intensity is independent. This is a significant speedup for the fitting loop.

```julia
# Parallel ensemble approach
function simulate_intensity_series_parallel(θ, intensities; t_total=5000.0, dt=0.1)
    col = build_column_from_params(θ)
    sidx = StateIndex(col)
    
    function prob_func(prob, i, repeat)
        col_i = @set col.stimulus.I_0 = intensities[i]
        u0 = dark_adapted_state(col_i, sidx)
        remake(prob; u0=u0, p=(col_i, sidx))
    end
    
    base_prob = ODEProblem(retinal_column_rhs!, 
                           dark_adapted_state(col, sidx), 
                           (0.0, t_total), (col, sidx))
    
    ensemble_prob = EnsembleProblem(base_prob; 
                                    prob_func=prob_func,
                                    safetycopy=true)
    
    ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads();
                         trajectories=length(intensities),
                         saveat=dt, abstol=1e-8, reltol=1e-6)
    
    # Extract ERG from each trajectory
    # ...
end
```

---

## 5. Stage 1: Point Estimation via Gradient-Based Optimization

### 5.1 Optimization Strategy

For each fitting stage, use a two-phase optimization:

**Phase 1: Global exploration** with multi-start or black-box optimization to find a good basin of attraction. This avoids getting trapped in a poor local minimum.

**Phase 2: Gradient-based refinement** using ADAM (robust to noisy/flat gradients) followed by L-BFGS (fast precise convergence near the minimum).

### 5.2 Gradient Computation: Adjoint Sensitivity

The key algorithmic choice: use **adjoint sensitivity analysis** to compute gradients of the loss with respect to parameters. This computes ∂L/∂θ in O(1) cost relative to the number of parameters (vs. O(N_params) for forward sensitivity). For our 48-parameter problem, this is a ~48x speedup.

```julia
using SciMLSensitivity

# When constructing the ODE problem for gradient-based fitting:
sensealg = InterpolatingAdjoint(
    autojacvec = ReverseDiffVJP(true),  # compiled tape for VJP
    checkpointing = true                 # O(1) memory
)

# The adjoint method is specified when solving:
sol = solve(prob, Tsit5(); 
            sensealg = sensealg,
            saveat = dt)
```

**Why `InterpolatingAdjoint`:** It stores the forward solution via interpolation and solves the adjoint backwards. With checkpointing, memory usage is O(√N) instead of O(N). The `ReverseDiffVJP(true)` option uses a compiled ReverseDiff tape for the vector-Jacobian products, which is faster than Zygote for in-place ODE functions.

**Fallback:** If adjoint methods have stability issues (possible with stiff systems), fall back to `ForwardDiffSensitivity(chunk_size=12)` which uses forward-mode AD with chunking. Less efficient for many parameters but more robust.

### 5.3 Implementation

```julia
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using OptimizationBBO, OptimizationMultistartOptimization
using SciMLSensitivity, ADTypes

"""
    fit_stage(stage, θ_init, data, features; 
              free_groups, frozen_params, kwargs...)

Fit parameters for one stage of the pipeline.

Arguments:
- stage: :photoreceptor, :on_pathway, :oscillatory, :slow_components, 
         :erg_weights, :global
- θ_init: initial parameter ComponentArray
- data: ERGDataSet
- features: ERGFeatures
- free_groups: which parameter groups to optimize (e.g., [:phototrans])
- frozen_params: fixed parameter values for groups not being optimized
"""
function fit_stage(stage::Symbol, θ_init, data, features;
                   free_groups, frozen_params=nothing,
                   use_global_search=true,
                   max_adam_iters=500,
                   max_lbfgs_iters=200,
                   verbose=true)
    
    # Extract free parameters
    θ_free = extract_free_params(θ_init, free_groups)
    bounds = get_bounds(free_groups)
    lb, ub = bounds.lower, bounds.upper
    
    # Build the loss function for this stage
    function loss_fn(θ_free_vec, p)
        θ_full = reconstruct_params(θ_free_vec, frozen_params, free_groups)
        sim_t, sim_erg = simulate_intensity_series(θ_full, data.intensities)
        sim_features = extract_erg_features_from_sim(sim_t, sim_erg)
        return stage_loss(stage, θ_full, (sim_t, sim_erg), data, sim_features)
    end
    
    # ──────────────────────────────────────────────
    # Phase 1: Global search (optional)
    # ──────────────────────────────────────────────
    if use_global_search
        if verbose println("  Phase 1: Global search...") end
        
        optf_global = OptimizationFunction(loss_fn)
        optprob_global = OptimizationProblem(optf_global, θ_free, nothing;
                                             lb=lb, ub=ub)
        
        # Multi-start with TikTak + L-BFGS (50 random starts)
        # Or use BBO for truly derivative-free global search
        result_global = solve(optprob_global, 
                              BBO_adaptive_de_rand_1_bin_radiuslimited();
                              maxiters=5000, maxtime=120.0)
        
        θ_free = result_global.u
        if verbose 
            println("    Global search loss: $(result_global.objective)")
        end
    end
    
    # ──────────────────────────────────────────────
    # Phase 2a: ADAM (gradient-based, robust)
    # ──────────────────────────────────────────────
    if verbose println("  Phase 2a: ADAM refinement...") end
    
    optf_adam = OptimizationFunction(loss_fn, ADTypes.AutoZygote())
    optprob_adam = OptimizationProblem(optf_adam, θ_free, nothing;
                                       lb=lb, ub=ub)
    
    callback_adam = build_callback(verbose, :adam)
    result_adam = solve(optprob_adam, 
                        Optimisers.Adam(0.001);
                        maxiters=max_adam_iters,
                        callback=callback_adam)
    
    θ_free = result_adam.u
    if verbose println("    ADAM loss: $(result_adam.objective)") end
    
    # ──────────────────────────────────────────────
    # Phase 2b: L-BFGS (gradient-based, precise)
    # ──────────────────────────────────────────────
    if verbose println("  Phase 2b: L-BFGS refinement...") end
    
    optprob_lbfgs = OptimizationProblem(optf_adam, θ_free, nothing;
                                         lb=lb, ub=ub)
    
    callback_lbfgs = build_callback(verbose, :lbfgs)
    result_lbfgs = solve(optprob_lbfgs, 
                          Optim.LBFGS();
                          maxiters=max_lbfgs_iters,
                          callback=callback_lbfgs,
                          allow_f_increases=false)
    
    θ_free = result_lbfgs.u
    if verbose println("    L-BFGS loss: $(result_lbfgs.objective)") end
    
    # Reconstruct full parameter vector
    θ_fitted = reconstruct_params(θ_free, frozen_params, free_groups)
    
    return θ_fitted, result_lbfgs.objective
end
```

### 5.4 Callback for Monitoring

```julia
function build_callback(verbose, phase)
    iter_count = Ref(0)
    best_loss = Ref(Inf)
    
    function callback(state, loss)
        iter_count[] += 1
        if loss < best_loss[]
            best_loss[] = loss
        end
        if verbose && (iter_count[] % 50 == 0)
            println("    [$phase] iter $(iter_count[]), loss = $(round(loss, digits=6)), best = $(round(best_loss[], digits=6))")
        end
        return false  # don't terminate early
    end
    
    return callback
end
```

---

## 6. Stage 2: Bayesian Inference via Turing.jl + NUTS

### 6.1 Why NUTS

The No-U-Turn Sampler (NUTS) is the gold standard for sampling continuous posterior distributions. It is a variant of Hamiltonian Monte Carlo (HMC) that automatically tunes the trajectory length, eliminating a key hyperparameter. NUTS is well-suited to this problem because:

1. It uses gradient information (efficient for 48 parameters)
2. It handles correlated parameters well (common in biophysical models)
3. The Julia implementation in Turing.jl composes directly with DifferentialEquations.jl
4. It provides well-calibrated posteriors with automatic convergence diagnostics

### 6.2 Turing Model Definition

```julia
using Turing, Distributions, DifferentialEquations

@model function erg_model(data::ERGDataSet, features::ERGFeatures,
                          base_prob, sidx, col_template;
                          fit_stage=:global)
    
    # ═══════════════════════════════════════════════
    # PRIORS (biophysically motivated)
    # ═══════════════════════════════════════════════
    
    # Observation noise
    σ_obs ~ InverseGamma(2, 30)
    
    # --- Group A: Phototransduction ---
    τ_R_rod ~ truncated(LogNormal(log(80), 0.3); lower=30, upper=200)
    τ_R_cone ~ truncated(LogNormal(log(10), 0.3); lower=3, upper=30)
    γ_rod ~ truncated(LogNormal(log(5), 0.5); lower=1, upper=20)
    γ_cone ~ truncated(LogNormal(log(3), 0.5); lower=0.5, upper=15)
    g_CNG_rod ~ truncated(LogNormal(log(20), 0.3); lower=5, upper=50)
    g_CNG_cone ~ truncated(LogNormal(log(30), 0.3); lower=10, upper=60)
    n_Ca ~ truncated(Normal(4, 1); lower=1, upper=6)
    g_H_rod ~ truncated(LogNormal(log(2), 0.5); lower=0.5, upper=8)
    τ_H ~ truncated(LogNormal(log(50), 0.3); lower=20, upper=100)
    η_rod ~ Beta(6.7, 3.3)
    α_G ~ truncated(LogNormal(log(20), 0.4); lower=5, upper=50)
    
    # --- Group B: ON-Pathway ---
    τ_mGluR6 ~ truncated(LogNormal(log(30), 0.3); lower=10, upper=80)
    g_TRPM1 ~ truncated(LogNormal(log(10), 0.4); lower=2, upper=30)
    g_Ca_on ~ truncated(LogNormal(log(4), 0.4); lower=1, upper=12)
    g_K_on ~ truncated(LogNormal(log(8), 0.3); lower=2, upper=20)
    φ_on ~ truncated(LogNormal(log(0.067), 0.4); lower=0.01, upper=0.2)
    V3_on ~ truncated(Normal(12, 8); lower=-10, upper=30)
    α_mGluR6 ~ truncated(LogNormal(0, 0.4); lower=0.2, upper=3)
    τ_Glu_on ~ truncated(LogNormal(log(5), 0.4); lower=1, upper=20)
    
    # --- Group C: Amacrine / OPs ---
    g_Ca_a2 ~ truncated(LogNormal(log(8), 0.3); lower=2, upper=20)
    g_K_a2 ~ truncated(LogNormal(log(12), 0.3); lower=4, upper=25)
    φ_a2 ~ truncated(LogNormal(log(0.2), 0.3); lower=0.05, upper=0.5)
    V3_a2 ~ truncated(Normal(-10, 6); lower=-25, upper=5)
    g_Ca_gaba ~ truncated(LogNormal(log(8), 0.3); lower=2, upper=20)
    φ_gaba ~ truncated(LogNormal(log(0.15), 0.3); lower=0.03, upper=0.4)
    τ_Gly ~ truncated(LogNormal(log(4), 0.3); lower=1, upper=10)
    τ_GABA ~ truncated(LogNormal(log(7), 0.3); lower=2, upper=15)
    g_Gly_gaba ~ truncated(LogNormal(log(10), 0.3); lower=2, upper=25)
    g_GABA_a2 ~ truncated(LogNormal(log(10), 0.3); lower=2, upper=25)
    g_Glu_a2 ~ truncated(LogNormal(log(8), 0.3); lower=2, upper=20)
    
    # --- Group D: Slow Components ---
    g_Kir_end ~ truncated(LogNormal(log(5), 0.4); lower=1, upper=15)
    g_Kir_stalk ~ truncated(LogNormal(log(2), 0.4); lower=0.5, upper=8)
    τ_K_diff ~ truncated(LogNormal(log(200), 0.5); lower=50, upper=1000)
    α_K ~ truncated(LogNormal(log(0.001), 0.5); lower=0.0001, upper=0.01)
    τ_RPE ~ truncated(LogNormal(log(3000), 0.3); lower=1000, upper=8000)
    g_K_apical ~ truncated(LogNormal(log(5), 0.4); lower=1, upper=15)
    α_K_RPE ~ truncated(LogNormal(log(0.001), 0.5); lower=0.0001, upper=0.01)
    
    # --- Group E: ERG Weights ---
    w_rod ~ truncated(LogNormal(0, 0.5); lower=0.1, upper=5)
    w_cone ~ truncated(LogNormal(log(0.5), 0.5); lower=0.05, upper=3)
    w_on ~ truncated(Normal(-2, 1.5); lower=-8, upper=-0.5)
    w_off ~ truncated(Normal(-1, 1); lower=-5, upper=-0.1)
    w_a2 ~ truncated(LogNormal(log(0.3), 0.5); lower=0.01, upper=2)
    w_gaba ~ truncated(LogNormal(log(0.3), 0.5); lower=0.01, upper=2)
    w_muller ~ truncated(Normal(-1.5, 1); lower=-5, upper=-0.1)
    w_rpe ~ truncated(Normal(-1, 1); lower=-5, upper=-0.1)
    
    # --- Group F: Sensitivity ---
    I_ref_rod ~ truncated(LogNormal(0, 0.5); lower=0.1, upper=10)
    I_ref_cone ~ truncated(LogNormal(log(100), 0.5); lower=10, upper=1000)
    
    # ═══════════════════════════════════════════════
    # FORWARD MODEL
    # ═══════════════════════════════════════════════
    
    # Pack parameters into the format expected by the ODE
    θ = pack_params(τ_R_rod, τ_R_cone, γ_rod, γ_cone, g_CNG_rod, g_CNG_cone,
                    n_Ca, g_H_rod, τ_H, η_rod, α_G,
                    τ_mGluR6, g_TRPM1, g_Ca_on, g_K_on, φ_on, V3_on, 
                    α_mGluR6, τ_Glu_on,
                    g_Ca_a2, g_K_a2, φ_a2, V3_a2, g_Ca_gaba, φ_gaba,
                    τ_Gly, τ_GABA, g_Gly_gaba, g_GABA_a2, g_Glu_a2,
                    g_Kir_end, g_Kir_stalk, τ_K_diff, α_K,
                    τ_RPE, g_K_apical, α_K_RPE,
                    w_rod, w_cone, w_on, w_off, w_a2, w_gaba, w_muller, w_rpe,
                    I_ref_rod, I_ref_cone)
    
    # Simulate at each intensity
    for k in 1:length(data.intensities)
        col_k = build_column_from_params(θ; intensity=data.intensities[k])
        u0 = dark_adapted_state(col_k, sidx)
        prob_k = remake(base_prob; u0=u0, p=(col_k, sidx))
        
        # Solve with error handling
        sol_k = solve(prob_k, Tsit5(); 
                      saveat=step(data.t),
                      abstol=1e-7, reltol=1e-5,
                      maxiters=500_000)
        
        # Check for failed solve
        if sol_k.retcode != :Success
            Turing.@addlogprob! -Inf
            return
        end
        
        # Compute simulated ERG
        erg_sim = compute_erg_trace(sol_k, col_k, sidx)
        
        # ═══════════════════════════════════════════
        # LIKELIHOOD
        # ═══════════════════════════════════════════
        
        # Gaussian likelihood at each time point
        for j in eachindex(data.t)
            data.traces[j, k] ~ Normal(erg_sim[j], σ_obs)
        end
    end
end
```

### 6.3 Running the Sampler

```julia
"""
    run_bayesian_inference(data, features; 
                           n_chains=4, n_samples=2000, n_warmup=1000,
                           init_params=nothing)

Run NUTS sampling on the ERG model.

Arguments:
- data: ERGDataSet
- features: ERGFeatures  
- n_chains: number of parallel MCMC chains (4 is standard)
- n_samples: samples per chain after warmup
- n_warmup: warmup/adaptation samples (NUTS tunes step size here)
- init_params: MAP estimate from point estimation (strongly recommended)

Returns: MCMCChains.Chains object with posterior samples
"""
function run_bayesian_inference(data, features;
                                n_chains=4, 
                                n_samples=2000,
                                n_warmup=1000,
                                init_params=nothing)
    
    # Build model
    col_template = build_retinal_column()
    sidx = StateIndex(col_template)
    base_prob = ODEProblem(retinal_column_rhs!, 
                           dark_adapted_state(col_template, sidx),
                           (0.0, maximum(data.t)),
                           (col_template, sidx))
    
    model = erg_model(data, features, base_prob, sidx, col_template)
    
    # NUTS sampler configuration
    sampler = NUTS(
        n_warmup,           # adaptation steps
        0.65;               # target acceptance rate (0.6-0.8 typical)
        max_depth=8         # max tree depth (reduce if too slow)
    )
    
    # Initialize at MAP estimate if provided
    if init_params !== nothing
        init = map_to_turing_init(init_params)
        chains = sample(model, sampler, MCMCThreads(), 
                        n_samples, n_chains;
                        init_params=fill(init, n_chains),
                        progress=true)
    else
        chains = sample(model, sampler, MCMCThreads(),
                        n_samples, n_chains;
                        progress=true)
    end
    
    return chains
end
```

### 6.4 Staged Bayesian Inference (Practical Compromise)

Running NUTS on all 48 parameters simultaneously may be slow. A practical compromise: run Bayesian inference **per stage** with fewer parameters free, then combine:

```julia
"""
    run_staged_bayesian(data, features, map_estimate;
                         samples_per_stage=1000)

Run Bayesian inference stage by stage, conditioning on MAP estimates
for parameters not in the current stage.

This is an approximation to the full joint posterior but is much more
computationally tractable (8-15 params per stage vs. 48 total).
"""
function run_staged_bayesian(data, features, map_estimate;
                              samples_per_stage=1000)
    results = Dict{Symbol, Any}()
    
    stages = [
        (:photoreceptor, [:phototrans]),
        (:on_pathway, [:on_pathway]),
        (:oscillatory, [:amacrine]),
        (:slow_components, [:slow]),
        (:erg_weights, [:erg_weights]),
    ]
    
    for (stage_name, groups) in stages
        println("Bayesian inference: Stage $stage_name ($(sum(length, groups)) params)")
        
        # Build stage-specific Turing model with only free params
        stage_model = build_stage_model(stage_name, groups, 
                                         map_estimate, data, features)
        
        sampler = NUTS(500, 0.65; max_depth=8)
        chains = sample(stage_model, sampler, MCMCThreads(),
                        samples_per_stage, 4; progress=true)
        
        results[stage_name] = chains
    end
    
    return results
end
```

### 6.5 Full Joint Posterior (If Computationally Feasible)

After staged Bayesian inference, optionally run a shorter full joint posterior:

```julia
function run_full_posterior(data, features, map_estimate;
                            n_samples=500, n_warmup=500, n_chains=4)
    # Initialize at MAP, run shorter chains
    # This refines correlations between parameter groups
    # that staged inference misses
    chains = run_bayesian_inference(data, features;
                                    n_chains=n_chains,
                                    n_samples=n_samples,
                                    n_warmup=n_warmup,
                                    init_params=map_estimate)
    return chains
end
```

---

## 7. Staged Fitting Pipeline

### 7.1 Full Pipeline

```julia
"""
    fit_retinal_model(data_path; kwargs...)

Main entry point for the full fitting pipeline.

Returns:
- map_estimate: best-fit parameters (ComponentArray)
- stage_results: per-stage optimization results
- posterior_chains: MCMC chains (if run_bayesian=true)
- diagnostics: convergence diagnostics and plots
"""
function fit_retinal_model(data_path; 
                            run_bayesian=true,
                            bayesian_mode=:staged,  # :staged, :full, :both
                            verbose=true,
                            kwargs...)
    
    # ═══════════════════════════════════════════════
    # LOAD AND PREPROCESS DATA
    # ═══════════════════════════════════════════════
    data = load_erg_data(data_path)
    data, features = preprocess_erg(data)
    
    if verbose
        println("Loaded $(length(data.intensities)) intensity levels")
        println("Time range: $(data.t[1]) to $(data.t[end]) ms")
        println("Features extracted: a-wave, b-wave, OPs")
    end
    
    # ═══════════════════════════════════════════════
    # STAGE A: PHOTORECEPTOR → a-wave
    # ═══════════════════════════════════════════════
    println("\n═══ Stage A: Fitting photoreceptor parameters to a-wave ═══")
    θ = build_parameter_vector()
    
    θ, loss_a = fit_stage(:photoreceptor, θ, data, features;
                          free_groups=[:phototrans],
                          use_global_search=true)
    
    println("  Stage A complete. Loss: $loss_a")
    
    # ═══════════════════════════════════════════════
    # STAGE B: ON-PATHWAY → b-wave
    # ═══════════════════════════════════════════════
    println("\n═══ Stage B: Fitting ON-pathway parameters to b-wave ═══")
    
    θ, loss_b = fit_stage(:on_pathway, θ, data, features;
                          free_groups=[:on_pathway],
                          use_global_search=true)
    
    println("  Stage B complete. Loss: $loss_b")
    
    # ═══════════════════════════════════════════════
    # STAGE C: AMACRINE → OPs
    # ═══════════════════════════════════════════════
    println("\n═══ Stage C: Fitting amacrine parameters to OPs ═══")
    
    θ, loss_c = fit_stage(:oscillatory, θ, data, features;
                          free_groups=[:amacrine],
                          use_global_search=true)
    
    println("  Stage C complete. Loss: $loss_c")
    
    # ═══════════════════════════════════════════════
    # STAGE D: SLOW COMPONENTS → P3 + c-wave
    # ═══════════════════════════════════════════════
    println("\n═══ Stage D: Fitting slow component parameters ═══")
    
    θ, loss_d = fit_stage(:slow_components, θ, data, features;
                          free_groups=[:slow],
                          use_global_search=true)
    
    println("  Stage D complete. Loss: $loss_d")
    
    # ═══════════════════════════════════════════════
    # STAGE E: ERG WEIGHTS → scaling
    # ═══════════════════════════════════════════════
    println("\n═══ Stage E: Fitting ERG weights ═══")
    
    θ, loss_e = fit_stage(:erg_weights, θ, data, features;
                          free_groups=[:erg_weights],
                          use_global_search=false)  # linear-ish, no global needed
    
    println("  Stage E complete. Loss: $loss_e")
    
    # ═══════════════════════════════════════════════
    # STAGE F: GLOBAL REFINEMENT
    # ═══════════════════════════════════════════════
    println("\n═══ Stage F: Global refinement (all parameters) ═══")
    
    all_groups = [:phototrans, :on_pathway, :amacrine, :slow, 
                  :erg_weights, :scaling]
    
    θ, loss_f = fit_stage(:global, θ, data, features;
                          free_groups=all_groups,
                          use_global_search=false,  # already near optimum
                          max_adam_iters=200,
                          max_lbfgs_iters=500)
    
    println("  Global refinement complete. Loss: $loss_f")
    
    map_estimate = θ
    
    # ═══════════════════════════════════════════════
    # BAYESIAN INFERENCE
    # ═══════════════════════════════════════════════
    posterior_chains = nothing
    
    if run_bayesian
        println("\n═══ Bayesian Inference ═══")
        
        if bayesian_mode == :staged || bayesian_mode == :both
            println("Running staged Bayesian inference...")
            staged_chains = run_staged_bayesian(data, features, map_estimate)
            posterior_chains = staged_chains
        end
        
        if bayesian_mode == :full || bayesian_mode == :both
            println("Running full joint posterior...")
            full_chains = run_full_posterior(data, features, map_estimate)
            posterior_chains = (bayesian_mode == :both) ? 
                (staged=staged_chains, full=full_chains) : full_chains
        end
        
        # Diagnostics
        print_convergence_diagnostics(posterior_chains)
    end
    
    # ═══════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════
    return (
        map_estimate = map_estimate,
        final_loss = loss_f,
        posterior = posterior_chains,
        data = data,
        features = features
    )
end
```

---

## 8. Interactive Mode

### 8.1 Interactive Fitting Interface

For debugging and exploration, provide functions that let Matt fix some parameters and explore others manually:

```julia
"""
    interactive_fit(data; kwargs...)

Launch an interactive fitting session. Provides functions for:
- Manually setting parameter values and viewing resulting ERG
- Running individual fitting stages
- Freezing/unfreezing parameter groups
- Comparing simulated vs observed traces
"""
function interactive_fit(data)
    θ = build_parameter_vector()
    data, features = preprocess_erg(data)
    
    # Return a mutable struct with convenience methods
    session = FittingSession(θ, data, features)
    return session
end

mutable struct FittingSession
    θ::ComponentArray
    data::ERGDataSet
    features::ERGFeatures
    frozen_groups::Set{Symbol}
    history::Vector{NamedTuple}  # track all parameter changes
end

# Set a single parameter
function set_param!(s::FittingSession, name::Symbol, value::Float64)
    # Navigate ComponentArray to set value
    set_nested!(s.θ, name, value)
    push!(s.history, (action=:set, param=name, value=value, 
                       loss=evaluate_loss(s)))
end

# Preview current fit
function preview(s::FittingSession; intensities=nothing)
    ints = isnothing(intensities) ? s.data.intensities : intensities
    sim_t, sim_erg = simulate_intensity_series(s.θ, ints)
    plot_comparison(s.data, sim_t, sim_erg)
end

# Freeze/unfreeze parameter groups
function freeze!(s::FittingSession, groups...)
    for g in groups
        push!(s.frozen_groups, g)
    end
end

function unfreeze!(s::FittingSession, groups...)
    for g in groups
        delete!(s.frozen_groups, g)
    end
end

# Run a single fitting stage
function run_stage!(s::FittingSession, stage::Symbol)
    free = setdiff(Set([:phototrans, :on_pathway, :amacrine, :slow, 
                        :erg_weights, :scaling]), s.frozen_groups)
    s.θ, loss = fit_stage(stage, s.θ, s.data, s.features;
                          free_groups=collect(free))
    push!(s.history, (action=:fit_stage, stage=stage, loss=loss))
    return loss
end

# Evaluate current loss without fitting
function evaluate_loss(s::FittingSession)
    sim_t, sim_erg = simulate_intensity_series(s.θ, s.data.intensities)
    sim_features = extract_erg_features_from_sim(sim_t, sim_erg)
    return full_loss(s.θ, (sim_t, sim_erg), s.data, sim_features)
end

# Sensitivity analysis: which parameters affect the loss most
function sensitivity(s::FittingSession; perturbation=0.1)
    base_loss = evaluate_loss(s)
    sensitivities = Dict{Symbol, Float64}()
    
    for (name, val) in pairs(flatten(s.θ))
        θ_pert = copy(s.θ)
        set_nested!(θ_pert, name, val * (1.0 + perturbation))
        pert_loss = evaluate_loss_with_params(θ_pert, s.data)
        sensitivities[name] = (pert_loss - base_loss) / (perturbation * abs(val))
    end
    
    # Sort by absolute sensitivity
    sorted = sort(collect(sensitivities), by=x->abs(x[2]), rev=true)
    return sorted
end
```

---

## 9. Diagnostics & Visualization

### 9.1 Point Estimation Diagnostics

```julia
"""
    plot_fit_result(result, data)

Generate a comprehensive diagnostic plot showing:
1. Simulated vs observed ERG at each intensity (overlay)
2. ERG component decomposition at one representative intensity
3. Intensity-response functions (a-wave, b-wave) with Naka-Rushton fits
4. OP analysis (filtered waveforms, frequency spectrum)
5. Residuals
"""
function plot_fit_result(result, data)
    fig = Figure(resolution=(1800, 1200))
    
    # Panel 1: Waterfall plot of ERG traces (sim vs obs)
    ax1 = Axis(fig[1, 1:2], title="ERG Intensity Series",
               xlabel="Time (ms)", ylabel="Amplitude (µV)")
    
    sim_t, sim_erg = simulate_intensity_series(result.map_estimate, 
                                                data.intensities)
    
    offsets = range(0, stop=500, length=length(data.intensities))
    for (k, I_k) in enumerate(data.intensities)
        lines!(ax1, data.t, data.traces[:, k] .+ offsets[k], 
               color=:black, linewidth=0.5)
        lines!(ax1, sim_t, sim_erg[:, k] .+ offsets[k], 
               color=:red, linewidth=1.0)
    end
    
    # Panel 2: Component decomposition
    ax2 = Axis(fig[1, 3], title="ERG Decomposition (mid intensity)")
    # ... plot a-wave, b-wave, OP, P3, c-wave contributions
    
    # Panel 3: Intensity-response functions
    ax3 = Axis(fig[2, 1], title="Intensity-Response",
               xlabel="Log Intensity", ylabel="Amplitude (µV)")
    # ... scatter + Naka-Rushton fit for a-wave and b-wave
    
    # Panel 4: OP analysis
    ax4 = Axis(fig[2, 2], title="Oscillatory Potentials",
               xlabel="Time (ms)", ylabel="Amplitude (µV)")
    # ... filtered OPs overlay
    
    # Panel 5: Residuals
    ax5 = Axis(fig[2, 3], title="Residuals",
               xlabel="Time (ms)", ylabel="Residual (µV)")
    
    return fig
end
```

### 9.2 Bayesian Diagnostics

```julia
"""
    print_convergence_diagnostics(chains)

Check MCMC convergence and print summary.
"""
function print_convergence_diagnostics(chains)
    # R-hat: should be < 1.01 for all parameters
    rhat = MCMCChains.rhat(chains)
    bad_rhat = filter(x -> x.rhat > 1.01, eachrow(rhat))
    
    if length(bad_rhat) > 0
        println("⚠️  Parameters with R-hat > 1.01 (not converged):")
        for row in bad_rhat
            println("    $(row.param): R-hat = $(round(row.rhat, digits=3))")
        end
    else
        println("✓ All parameters have R-hat < 1.01")
    end
    
    # Effective sample size: should be > 100 per chain
    ess = MCMCChains.ess(chains)
    bad_ess = filter(x -> x.ess < 400, eachrow(ess))  # 100 per chain × 4
    
    if length(bad_ess) > 0
        println("⚠️  Parameters with low ESS (< 400):")
        for row in bad_ess
            println("    $(row.param): ESS = $(round(row.ess, digits=0))")
        end
    else
        println("✓ All parameters have ESS > 400")
    end
    
    # Summary statistics
    println("\nPosterior Summary (median [95% CI]):")
    for param in names(chains, :parameters)
        q = quantile(chains[param], [0.025, 0.5, 0.975])
        println("  $param: $(round(q[2], sigdigits=4)) [$(round(q[1], sigdigits=3)), $(round(q[3], sigdigits=3))]")
    end
end

"""
    plot_posterior(chains; params=nothing)

Generate posterior diagnostic plots:
1. Marginal distributions with prior overlay
2. Trace plots
3. Corner plot for correlated parameter pairs
4. Posterior predictive check (simulated ERG ensemble)
"""
function plot_posterior(chains; params=nothing)
    # ... GLMakie-based plotting
end
```

### 9.3 Posterior Predictive Check

The most important diagnostic: sample parameters from the posterior and simulate ERG traces. The spread of these traces should bracket the observed data.

```julia
"""
    posterior_predictive(chains, data; n_draws=100)

Draw parameter sets from the posterior and simulate ERG.
Returns ensemble of ERG traces for visual comparison with data.
"""
function posterior_predictive(chains, data; n_draws=100)
    n_total = size(chains, 1) * size(chains, 3)  # samples × chains
    draw_indices = rand(1:n_total, n_draws)
    
    erg_ensemble = []
    
    for idx in draw_indices
        θ_draw = extract_params_from_chain(chains, idx)
        sim_t, sim_erg = simulate_intensity_series(θ_draw, data.intensities)
        push!(erg_ensemble, sim_erg)
    end
    
    return erg_ensemble
end
```

---

## 10. Code Architecture

### 10.1 Module Structure (Additions to Base Project)

```
RetinalTwin/
├── src/
│   ├── ...                        # (existing modules from base spec)
│   ├── fitting/
│   │   ├── parameters.jl          # Parameter vector, bounds, transforms
│   │   ├── data.jl                # ERGDataSet, loading, preprocessing
│   │   ├── features.jl            # Feature extraction, Naka-Rushton
│   │   ├── loss.jl                # All loss functions
│   │   ├── point_estimation.jl    # Optimization pipeline
│   │   ├── bayesian.jl            # Turing model, NUTS sampling
│   │   ├── pipeline.jl            # Staged fitting orchestration
│   │   ├── interactive.jl         # FittingSession for exploration
│   │   └── diagnostics.jl         # Plotting and convergence checks
│   └── RetinalTwin.jl             # Updated exports
├── test/
│   ├── ...                        # (existing tests)
│   ├── test_fitting.jl            # Synthetic data recovery tests
│   └── test_bayesian.jl           # Prior predictive, posterior checks
└── Project.toml                   # Updated dependencies
```

---

## 11. Implementation Roadmap

### Week 1: Data Interface + Loss Functions
1. Implement `ERGDataSet` struct and data loading (CSV, MAT)
2. Implement `preprocess_erg` (baseline subtraction, filtering)
3. Implement `extract_erg_features` (a-wave, b-wave, OPs, Naka-Rushton)
4. Implement all loss functions (waveform, feature, stage-specific)
5. Write synthetic data tests (generate known ERG, verify feature extraction)

### Week 2: Point Estimation Pipeline
1. Implement `build_parameter_vector` with ComponentArrays
2. Implement parameter group extraction and reconstruction
3. Implement `fit_stage` with BBO → ADAM → L-BFGS pipeline
4. Wire up adjoint sensitivity (`InterpolatingAdjoint`)
5. Implement `simulate_intensity_series` with parallel ensemble support
6. Test on synthetic data: generate ERG with known params, recover them

### Week 3: Bayesian Inference
1. Implement Turing `@model` for full and staged models
2. Implement `run_bayesian_inference` with NUTS
3. Implement `run_staged_bayesian` 
4. Implement convergence diagnostics (R-hat, ESS)
5. Implement posterior predictive checks
6. Test on synthetic data: verify posterior covers true parameters

### Week 4: Interactive Mode + Polish
1. Implement `FittingSession` for interactive exploration
2. Implement `sensitivity` analysis
3. Build comprehensive diagnostic plotting
4. Write documentation and usage examples
5. Performance optimization (profiling, reducing allocations)
6. Test with Matt's actual ERG data

---

## 12. Dependencies

Add to `Project.toml`:

```toml
[deps]
# Existing
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7caa7"
ComponentArrays = "b0b7db55-cfe3-40fc-9bbe-47749f4d276d"

# Optimization
Optimization = "7f7a1694-90dd-40f0-9382-eb1880f1de58"
OptimizationOptimisers = "42dfb2eb-d2b4-4451-abcd-913932933ac1"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
OptimizationBBO = "3e6eede4-6e63-4e54-8b04-7a7acc90dfe2"
OptimizationMultistartOptimization = "e4316d97-8f59-4d6a-b1bd-a882a1a2dab7"

# Automatic Differentiation
SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"

# Bayesian
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"

# Signal processing  
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"

# Data handling
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MAT = "23992714-dd62-5051-b70f-ba4f8e5a196f"

# Visualization
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"

# Utilities
Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
```

---

## Appendix: Quick Reference for Claude Code

### Key Decision Summary

| Decision | Choice | Reason |
|----------|--------|--------|
| Gradient method | Adjoint sensitivity (InterpolatingAdjoint) | O(1) in N_params for 48 params |
| AD backend | ReverseDiffVJP(true) for ODE, Zygote for outer loop | Compiled tape handles in-place mutation |
| Global optimizer | BBO (adaptive DE) | Robust derivative-free exploration |
| Local optimizer | ADAM → L-BFGS | ADAM for noisy landscape, L-BFGS for precision |
| Bayesian sampler | NUTS via Turing.jl | Gold standard, auto-tunes, uses gradients |
| Fitting strategy | 5-stage decomposition + global refinement | Exploits known cell→ERG mapping |
| Parameter transforms | Log for positive, logit for bounded | Ensures unconstrained optimization |
| Multi-intensity | EnsembleProblem with threading | Independent solves, easy parallelism |
| Posterior mode | Staged (8-15 params each) then optional full joint | Tractable for 48 total params |

### If Things Go Wrong

1. **ODE solver fails during optimization:** Add `try/catch` around `solve()`, return `Inf` loss on failure. Consider switching to `TRBDF2()` (implicit) if stiffness is the issue.
2. **Adjoint sensitivity unstable:** Fall back to `ForwardDiffSensitivity(chunk_size=12)`. Slower but more robust.
3. **NUTS slow/divergent:** Reduce `max_depth` to 6, increase target acceptance to 0.8, or use staged Bayesian only.
4. **Loss stuck in local minimum:** Increase BBO iterations, try `MultistartOptimization.TikTak(200)` with more starting points.
5. **Feature extraction fails on noisy data:** Smooth data with Savitzky-Golay filter before feature extraction. Increase temporal weighting on waveform loss relative to feature loss.
