# RetinalTwin.jl

A biophysically grounded mechanistic simulation of the vertebrate retinal neural circuit implemented in Julia.

## Overview

RetinalTwin.jl is a computational model of a single retinal column—the minimal vertical pathway from photoreceptors through bipolar cells to ganglion cells. Unlike machine learning black boxes, every parameter in this model has direct biological meaning: ion channel conductances, receptor kinetics, synaptic weights, and signaling cascade time constants. The model outputs simulated electroretinogram (ERG) waveforms directly comparable to clinical recordings.

## Current Implementation Status

### Cell Types (10 implemented)
- **Photoreceptors**: Rod with detailed phototransduction cascade (21 ODEs)
- **Horizontal cells**: Lateral inhibition network (3 ODEs)
- **ON bipolar cells**: mGluR6 metabotropic sign-inverting synapse (4 ODEs)
- **OFF bipolar cells**: Ionotropic glutamate receptors (4 ODEs)
- **A2 amacrine cells**: Glycinergic, oscillatory potential generation (3 ODEs)
- **GABAergic amacrine cells**: Reciprocal inhibition network (3 ODEs)
- **Dopaminergic amacrine cells**: Slow modulatory signaling (3 ODEs)
- **Ganglion cells**: Action potential generation and retinal output (2 ODEs)
- **Müller glia**: K⁺ buffering and glutamate uptake (4 ODEs)
- **Retinal pigment epithelium (RPE)**: c-wave generation via K⁺ sensing (2 ODEs)

### Model Architecture
```
State Variables per Cell Type:
├── Photoreceptor: 21 (R, T, P, cGMP, Ih gating (5), ion channels (4), Ca²⁺ dynamics (6), V, Glu)
├── Horizontal: 3 (V, w, Glu tracking)
├── ON Bipolar: 4 (V, w, mGluR6 state, Glu release)
├── OFF Bipolar: 4 (V, w, iGluR gating, Glu release)
├── A2 Amacrine: 3 (V, w, glycine release)
├── GABA Amacrine: 3 (V, w, GABA release)
├── DA Amacrine: 3 (V, w, dopamine release)
├── Ganglion: 2 (V, w)
├── Müller: 4 (V, K⁺ endfoot, K⁺ stalk, Glu uptake)
└── RPE: 2 (V, subretinal K⁺)

Total per complete column: 49 ODEs
Current working example: 25 ODEs (photoreceptor + ON bipolar)
```

### Biophysical Parameters
Approximately **197 interpretable parameters** across all cell types:
- **Photoreceptor**: 59 parameters (phototransduction kinetics, ion channel conductances, Ca²⁺ buffering, glutamate release)
- **Bipolar cells**: 20 (ON) + 19 (OFF) parameters (Morris-Lecar dynamics, receptor kinetics)
- **Horizontal**: 16 parameters
- **Amacrine cells**: 16 (A2) + 16 (GABA) + 16 (DA) parameters
- **Ganglion**: 16 parameters
- **Müller glia**: 9 parameters (Kir channels, EAAT transporters)
- **RPE**: 10 parameters (K⁺ handling, slow time constants)

All parameters stored in CSV files (`src/parameters/`) with bounds for future Bayesian fitting.

## Platform Architecture

### Layer 1: The Digital Organ (RetinalTwin.jl)

A mechanistic simulation implemented in Julia comprising **49 coupled ODEs** across **10 retinal cell types** (rods, horizontal, ON/OFF-bipolar, A2/GABAergic/dopaminergic amacrine, ganglion, Müller glia, RPE) governed by **~197 biophysical parameters** with direct biological meaning.

#### Key Biophysical Mechanisms

**Photoreceptor (21 ODEs, 59 parameters)**:
- Simplified phototransduction cascade: rhodopsin activation → transducin → phosphodiesterase → cGMP hydrolysis
- Cyclic nucleotide-gated (CNG) channels with cooperative gating
- 5-state hyperpolarization-activated current (Ih) for temporal filtering
- Voltage-gated K⁺ (Kv), L-type Ca²⁺, Ca²⁺-activated K⁺ (KCa), and Ca²⁺-activated Cl⁻ currents
- Detailed inner segment Ca²⁺ dynamics with fast/slow compartments and dual-affinity buffers
- Graded voltage-dependent glutamate release (tau = 10 ms)

**ON Bipolar (4 ODEs, 20 parameters)**:
- mGluR6 metabotropic cascade with sign inversion: high Glu → TRPM1 closed → hyperpolarization
- Morris-Lecar excitability for graded depolarization
- Setting `g_TRPM1 → 0` produces electronegative ERG (congenital stationary night blindness phenotype)

**OFF Bipolar (4 ODEs, 19 parameters)**:
- Ionotropic glutamate receptors (iGluR) with fast kinetics (tau ~ 3 ms)
- Sign-preserving: high Glu → depolarization

**Amacrine Network (3 cell types, 9 ODEs total)**:
- A2 (AII) and GABAergic amacrines form reciprocal inhibitory loops for oscillatory potential (OP) generation (100-160 Hz)
- Dopaminergic amacrines provide slow modulatory input (tau ~ 200 ms)

**Müller Glia (4 ODEs, 9 parameters)**:
- Inward-rectifying K⁺ channels (Kir) at endfoot and stalk for K⁺ spatial buffering
- Excitatory amino acid transporters (EAAT) for glutamate uptake
- Generates P3 component of ERG via transmembrane K⁺ currents

**RPE (2 ODEs, 10 parameters)**:
- Apical K⁺ channels sense photoreceptor dark current
- Light → reduced photoreceptor K⁺ efflux → subretinal [K⁺] drops → RPE hyperpolarizes
- Generates slow c-wave component (peak 2-5 seconds)

#### ERG Output
Simulated ERG is computed as weighted sum of transmembrane currents from photoreceptors, bipolar cells, Müller glia, and RPE. The model can reproduce:
- **a-wave**: Photoreceptor hyperpolarization (negative, onset <20 ms)
- **b-wave**: ON bipolar depolarization (positive, onset 30-60 ms)
- **c-wave**: RPE slow response (positive, peak 2-5 s)
- **Oscillatory potentials (OPs)**: Amacrine network oscillations (75-300 Hz)

#### Parameter Interpretability Example
```julia
# Simulate TRPM1 mutation (ON bipolar channelopathy)
params.ON_BIPOLAR_PARAMS.g_TRPM1 = 0.0  # nS

# Result: Electronegative ERG
# - No b-wave (ON bipolar pathway abolished)
# - Preserved a-wave (photoreceptors intact)
# - Matches clinical CSNB (congenital stationary night blindness) phenotype
```

### Layer 2: Parameter Estimation (In Development)

**Staged Bayesian Inference Pipeline**:
Exploits known ERG-to-cell-type mapping to fit parameters sequentially:
1. **Stage 1**: Photoreceptor parameters from isolated a-wave (first 30 ms, before b-wave onset)
2. **Stage 2**: Bipolar parameters from b-wave amplitude and kinetics
3. **Stage 3**: Amacrine parameters from oscillatory potential frequency and amplitude
4. **Stage 4**: Müller/RPE parameters from slow components (P3, c-wave)

**Sampling Strategy**:
- NUTS (No-U-Turn Sampler) for efficient posterior exploration
- Outputs: Point estimates + full posterior distributions for uncertainty quantification
- Priors: Informative bounds from literature (stored in CSV `LowerBounds`, `UpperBounds` columns)

### Layer 3: Disease Modeling (Planned)

**Disease Perturbation Framework**:
- Parameter knockdowns map directly to genetic mutations
- Examples:
  - `TRPM1 → 0`: Complete congenital stationary night blindness (cCSNB)
  - `Rhodopsin kinetics ↓`: Retinitis pigmentosa
  - `mGluR6 cascade ↓`: Incomplete cCSNB
  - `Photoreceptor degeneration`: Age-related macular degeneration (reduce N_rod, N_cone)

**Planned Validation Data Sources**:
- Clinical ERG databases (XLRS, RP, CSNB patient cohorts)
- Lab recordings from genetic knockout models (CRISPR, transgenics)
- Single-cell transcriptomic atlases for parameter constraints

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/RetinalTwin.jl")
```

**Dependencies**:
- `DifferentialEquations.jl`: ODE solver (Rodas5 for stiff systems)
- `CairoMakie.jl`: Visualization
- `CSV.jl`, `DataFrames.jl`: Parameter management

## Quick Start

### Minimal Example: Photoreceptor → ON Bipolar

```julia
using RetinalTwin
using DifferentialEquations
using CairoMakie

# Load default parameters
params = default_retinal_params()

# Build initial conditions (dark-adapted state)
u0 = retinal_column_initial_conditions(params)

# Define light stimulus
stim_params = (
    stim_start = 50.0,    # ms
    stim_end = 55.0,      # 5 ms flash
    photon_flux = 10.0,   # photons/µm²/ms
)

# Solve ODE system
tspan = (0.0, 300.0)  # ms
prob = ODEProblem(retinal_column_model!, u0, tspan, (params, stim_params))
sol = solve(prob, Rodas5(); saveat=1.0, abstol=1e-6, reltol=1e-4)

# Extract results
t = sol.t
V_photo = [u[RetinalTwin.ROD_V_INDEX] for u in sol.u]
V_onbc = [u[22] for u in sol.u]

# Plot
fig = Figure()
ax = Axis(fig[1,1], xlabel="Time (ms)", ylabel="V (mV)")
lines!(ax, t, V_photo, label="Photoreceptor")
lines!(ax, t, V_onbc, label="ON Bipolar")
axislegend(ax)
display(fig)
```

### Output
- Light → Photoreceptor hyperpolarizes (~-40 → -50 mV)
- Hyperpolarization → Glutamate release drops
- Low glutamate → mGluR6 cascade decreases
- TRPM1 opens → ON bipolar **depolarizes** (~-60 → -45 mV)
- **Sign-inverted response achieved!**

## Project Structure

```
RetinalTwin.jl/
├── src/
│   ├── RetinalTwin.jl           # Module entry point
│   ├── types.jl                 # Core type definitions
│   ├── parameters/
│   │   ├── parameter_extraction.jl        # CSV loading functions
│   │   ├── photoreceptor_params.csv       # 59 parameters
│   │   ├── on_bipolar_params.csv          # 20 parameters
│   │   ├── off_bipolar_params.csv         # 19 parameters
│   │   ├── horizontal_params.csv          # 16 parameters
│   │   ├── a2_amacrine_params.csv         # 16 parameters
│   │   ├── gaba_amacrine_params.csv       # 16 parameters
│   │   ├── da_amacrine_params.csv         # 16 parameters
│   │   ├── ganglion_params.csv            # 16 parameters
│   │   ├── muller_params.csv              # 9 parameters
│   │   └── rpe_params.csv                 # 10 parameters
│   ├── stimulus/
│   │   └── light.jl             # Stimulus protocols (flash, step, chirp)
│   ├── cells/
│   │   ├── photoreceptor.jl     # 21 ODEs: phototransduction + membrane
│   │   ├── horizontal.jl        # 3 ODEs: Morris-Lecar
│   │   ├── on_bipolar.jl        # 4 ODEs: ML + mGluR6
│   │   ├── off_bipolar.jl       # 4 ODEs: ML + iGluR
│   │   ├── a2_amacrine.jl       # 3 ODEs: ML + glycine
│   │   ├── gaba_amacrine.jl     # 3 ODEs: ML + GABA
│   │   ├── da_amacrine.jl       # 3 ODEs: ML + dopamine
│   │   ├── ganglion.jl          # 2 ODEs: ML
│   │   ├── muller.jl            # 4 ODEs: K⁺ buffering + Glu uptake
│   │   └── rpe.jl               # 2 ODEs: K⁺ sensing
│   ├── circuit/
│   │   └── retinal_column.jl    # Column assembly + coupling
│   ├── erg/
│   │   └── field_potential.jl   # ERG calculation
│   ├── simulation/
│   │   ├── ode_system.jl        # ODE right-hand side
│   │   └── run.jl               # Simulation driver
│   ├── validation/
│   │   └── targets.jl           # ERG metrics (a/b-wave, OP)
│   └── visualization/
│       ├── plots.jl             # 2D plotting
│       └── gl_plots.jl          # 3D/interactive
├── examples/
│   ├── run_simple_retinal_column.jl  # Photoreceptor + ON bipolar demo
│   ├── run_photoreceptor_single_flash.jl
│   └── run_model.jl             # Full retinal column (when complete)
├── docs/
│   ├── retinal_digital_twin_spec.md    # Full mathematical specification
│   ├── rod_photoreceptor_model.md
│   └── retinal_twin_fitting_spec.md
└── PROGRESS.md                  # Development tracker
```

## Design Principles

1. **Interpretability**: Every parameter corresponds to a measurable biophysical quantity
2. **Modularity**: Cell models are independent and can be validated in isolation
3. **Staged validation**: Fit photoreceptor → bipolar → amacrine → glia sequentially
4. **No arbitrary constraints**: Removed `max()`, `min()`, `clamp()` to avoid unbiological bounds
5. **CSV-driven parameters**: Easy to modify, version control, and share parameter sets

## Validation Strategy

### Phase 1: Single-Cell Validation
- [x] Photoreceptor dim flash: single photon response (SPR) ~1 pA, peak ~200 ms
- [x] Photoreceptor saturating flash: deep hyperpolarization + recovery
- [x] ON bipolar sign inversion: depolarizes when glutamate drops
- [ ] A2-GABA oscillations: 100-160 Hz
- [ ] Full ERG waveform: a/b/c-waves + OPs

### Phase 2: Parameter Fitting
- [ ] Fit to human ERG database (normal controls)
- [ ] Fit to disease ERG (CSNB, RP, XLRS)
- [ ] Cross-validate on held-out recordings

### Phase 3: Spatial Extension
- [ ] 2D grid of retinal columns for spatial ERG patterns
- [ ] Receptive field dynamics
- [ ] Lateral propagation (gap junctions, diffusion)

## Experimental Biology Integration (Planned)

**Closed-Loop Validation Pipeline**:
1. Record ERG from patient/animal model
2. Fit RetinalTwin.jl parameters via Bayesian inference
3. Model predicts most likely genetic/pharmacological perturbations
4. Validate predictions against:
   - scRNA-seq data
   - CRISPR knockout phenotypes
   - Clinical outcomes

**Active Learning**:
- Model identifies parameters with highest posterior uncertainty
- Proposes experiments that maximally reduce uncertainty
- Example: "Test photoreceptor Ca²⁺ buffering via pharmacological block to resolve `Hb1` vs `Lb1` ambiguity"

## Current Limitations

- **Single cell per type**: No spatial structure yet (planned for Phase 3)
- **No cone pathway**: Rod-only implementation (cones planned)
- **Simplified phototransduction**: Uses reduced cascade (Forti et al. model)
- **No gap junctions**: A2-cone bipolar electrical coupling not implemented
- **Idealized ERG**: No noise, recording electrode geometry, or volume conduction

## Comparison to Original Specification

The original fellowship application described a **193-ODE model** with populations of cells (20 rods, 5 cones, etc.). The current implementation uses a **single-cell retinal column** (49 ODEs for full model, 25 ODEs for working photoreceptor+ON-bipolar example). This design choice prioritizes:
1. **Faster iteration**: Single-cell models easier to debug and validate
2. **Parameter identifiability**: Fewer free parameters, tighter constraints
3. **Modular expansion**: Can scale to populations after validating single-cell dynamics

## References

1. **Phototransduction**: Forti et al. (1989), Lamb & Pugh (1992)
2. **Bipolar mGluR6**: Shiells & Falk (1990), Sampath & Rieke (2004)
3. **Morris-Lecar**: Morris & Lecar (1981), Rinzel & Ermentrout (1998)
4. **ERG origins**: Robson & Frishman (2014), "The origin of the electroretinogram"
5. **Müller K⁺ buffering**: Newman et al. (1984), Kofuji & Newman (2004)
6. **Oscillatory potentials**: Wachtmeister (1998)

## License

MIT

## Citation

If you use RetinalTwin.jl in your research, please cite:

```bibtex
@software{retinaltwin2026,
  author = {Tarchick, Matthew},
  title = {RetinalTwin.jl: A Biophysically Grounded Mechanistic Model of the Retinal Circuit},
  year = {2026},
  url = {https://github.com/yourusername/RetinalTwin.jl}
}
```

## Contact

Dr. Matt Tarchick
[Your contact information]
