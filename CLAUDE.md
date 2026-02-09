# CLAUDE.md -- Retinal Digital Twin Progress Tracker

## Project: RetinalTwin.jl
## Spec: retinal_digital_twin_spec.md

## Current Phase: Phase 1 -- Core Retinal Column

## Architecture
- Julia package: RetinalTwin
- ODE system: 193 state variables (flat vector), solved with DifferentialEquations.jl
- Cell types: Photoreceptors (rod/cone), HC, ON-BC, OFF-BC, A2, GABA-AC, DA-AC, GC, Muller, RPE
- Synaptic coupling: E/I/M framework with 19 defined connections
- ERG: weighted sum of transmembrane currents

## File Status

| File | Status | Notes |
|------|--------|-------|
| Project.toml | [x] | Package dependencies |
| src/RetinalTwin.jl | [x] | Module entry point |
| src/types.jl | [x] | Core type definitions |
| src/parameters.jl | [x] | Default parameter sets |
| src/stimulus/light.jl | [x] | Stimulus protocols |
| src/cells/photoreceptor.jl | [x] | Phototransduction + membrane |
| src/cells/horizontal.jl | [x] | HC dynamics |
| src/cells/on_bipolar.jl | [x] | mGluR6 sign-inverting |
| src/cells/off_bipolar.jl | [x] | Ionotropic bipolar |
| src/cells/a2_amacrine.jl | [x] | Glycinergic, OP generator |
| src/cells/gaba_amacrine.jl | [x] | GABAergic, OP generator |
| src/cells/da_amacrine.jl | [x] | Dopaminergic modulation |
| src/cells/ganglion.jl | [x] | Output neuron |
| src/cells/muller.jl | [x] | K+ buffering, P3 |
| src/cells/rpe.jl | [x] | c-wave generator |
| src/synapses/synapse.jl | [x] | Synaptic transmission |
| src/synapses/mglur6.jl | [x] | Sign-inverting synapse |
| src/synapses/modulatory.jl | [x] | Dopamine modulation |
| src/circuit/connectivity.jl | [x] | Connection matrix |
| src/circuit/retinal_column.jl | [x] | Assembly + initial conditions |
| src/erg/field_potential.jl | [x] | ERG calculation + OP extraction |
| src/simulation/ode_system.jl | [x] | Full ODE RHS |
| src/simulation/run.jl | [x] | Simulation driver |
| src/visualization/plots.jl | [x] | Plotting utilities |
| examples/run_model.jl | [x] | Interactive demo |

## Implementation Order
1. types.jl + parameters.jl (foundation)
2. stimulus/light.jl (needed by photoreceptors)
3. cells/*.jl (all 10 cell files)
4. synapses/*.jl (coupling framework)
5. circuit/*.jl (wiring + initial conditions)
6. simulation/*.jl (ODE assembly + driver)
7. erg/field_potential.jl (ERG computation)
8. visualization/plots.jl (plotting)
9. examples/run_model.jl (demo)
10. Test and validate

## Key Design Decisions
- Package name: RetinalTwin (repo dir stays VisualPathwayTwin.jl)
- CairoMakie for visualization (headless-compatible)
- NTReleaseParams struct for neurotransmitter release parameters
- PopulationSizes struct for cleaner RetinalColumn
- Synaptic currents pre-computed in ODE RHS, passed to cell update functions
- PhototransductionParams includes membrane params (g_L, E_L, C_m)

## Validation Targets (Phase 1)
- [ ] Photoreceptor dim flash: ~1 pA SPR, peak ~200 ms
- [ ] Photoreceptor saturating flash: deep hyperpolarization + nose
- [ ] ON-bipolar sign inversion: depolarizes when glutamate drops
- [ ] A2-GABA oscillations: 100-160 Hz
- [ ] ERG a-wave: negative, onset <20 ms
- [ ] ERG b-wave: positive, onset 30-60 ms
- [ ] ERG OPs: visible 75-300 Hz band
- [ ] c-wave: slow positive, peak 2-5 s
- [ ] Intensity-response: Naka-Rushton fit

## Next Steps After Phase 1
- Parameter fitting (Optimization.jl / BlackBoxOptim.jl)
- Noise model (Poisson photon noise, channel noise)
- Spatial grid expansion (Phase 3)
- Species-specific parameter sets
