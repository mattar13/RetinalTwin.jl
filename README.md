# Visual Pathway Digital Twin

A computational model of the visual pathway from retina to cortex, implementing Morris-Lecar neurons with excitatory (E), inhibitory (I), and modulatory (M) neurotransmitter systems.

## Overview

Based on **Tarchick et al. 2023** (*Scientific Reports*): "Modeling cholinergic retinal waves: starburst amacrine cells shape wave generation, propagation, and direction bias"

### Key Features

- **Morris-Lecar neuron model** with Hodgkin-Huxley sodium channels
- **E/I/M neurotransmitter systems**:
  - Excitatory (E): Glutamate/ACh via AMPA/nAChR
  - Inhibitory (I): GABA via GABA-A receptors
  - Modulatory (M): Metabotropic cascade → second messenger → ion channel
- **Modulatory cascade**: At (logistic growth) → Bt (second messenger) → iMod (tunable reversal E_MOD)
- **Discrete synaptic model**: No PDEs, neighboring neurons share NT directly
- **Complete visual pathway**: Retina → LGN → V1

## Cell Types

| Region | Cell Type | Primary NT | Key Properties |
|--------|-----------|------------|----------------|
| Retina | Bipolar | Excitatory (Glu) | Graded responses, strong Ca2+ |
| Retina | Ganglion (RGC) | Excitatory (Glu) | Spiking, projects to LGN |
| LGN | Thalamic Relay | Excitatory (Glu) | Burst/tonic modes, T-type Ca2+ |
| LGN | Thalamic Inhibitory | Inhibitory (GABA) | Fast-spiking, gain control |
| V1 | Cortical Pyramidal | Excitatory (Glu) | Adaptation, complex integration |
| V1 | Cortical Inhibitory | Inhibitory (GABA) | PV+ fast-spiking, perisomatic |
| V1 | Cortical Modulatory | Modulatory (GABA + peptides) | SOM+/VIP+, dendritic targeting |

## Model Equations

### Voltage Equation (Morris-Lecar extended)

```
C_m * dV/dt = I_leak + I_Ca + I_K + I_Na + I_TREK + I_MOD + I_E + I_I + I_noise + I_app
```

### Modulatory Cascade

1. **Logistic growth (At)**: Activated by Ca2+ and modulatory NT
   ```
   τ_A * dA/dt = α_A * (Ca^4 + H_M(M_mod)) * (1 - A) - A
   ```

2. **Second messenger (Bt)**: Driven by At with 4th-order kinetics
   ```
   τ_B * dB/dt = β_B * A^4 * (1 - B) - B
   ```

3. **Modulatory current (iMod)**: Channel gated by Bt
   ```
   I_MOD = -g_MOD * B * (V - E_MOD)
   ```

### Tunable Parameters

- **E_MOD**: Reversal potential of modulatory channel
  - E_MOD < E_rest: Inhibitory modulation
  - E_MOD ≈ E_rest: Shunting modulation
  - E_MOD > E_rest: Excitatory modulation

## Installation

```julia
using Pkg
Pkg.activate("VisualPathwayTwin")
Pkg.instantiate()
```

## Usage

```julia
include("src/VisualPathwayTwin.jl")
using .VisualPathwayTwin

# Create network
pathway = create_default_pathway(size=:medium)

# Run simulation
times, recordings = simulate!(pathway, 500.0)  # 500 ms

# Apply retinal stimulus
stimulus = create_stimulus_pattern(...)
times, rec = stimulate_retina!(pathway, stimulus; duration=200.0)

# Tune modulatory reversal
set_modulatory_reversal!(pathway, "CorticalPyramidal", -50.0)  # Depolarizing
```

## File Structure

```
VisualPathwayTwin/
├── Project.toml           # Julia project file
├── README.md              # This file
├── src/
│   ├── VisualPathwayTwin.jl  # Main module
│   ├── Neurons.jl            # Core neuron model
│   ├── CellTypes.jl          # Cell-type specific parameters
│   └── Networks.jl           # Network connectivity
├── examples/
│   └── run_simulation.jl     # Example usage
└── test/
    └── runtests.jl           # Unit tests
```

## Parameter Optimization

Each cell type has ~50 parameters that can be optimized. Key parameters for fitting:

- **Membrane**: C_m, g_leak, E_leak
- **Ionic channels**: g_Ca, g_K, g_Na, V1-V4 (gating)
- **sAHP/Modulation**: g_TREK, g_MOD, E_MOD, τ_A, τ_B
- **Synaptic**: g_E, g_I, g_M, τ_E, τ_I, τ_M

Recommended optimization approach:
1. Fit single-cell dynamics to patch-clamp data
2. Fit network dynamics to population recordings
3. Use genetic algorithms (as in Tarchick et al. 2023)

## References

- Tarchick MJ, Clute DA, Renna JM (2023). Modeling cholinergic retinal waves: starburst amacrine cells shape wave generation, propagation, and direction bias. *Scientific Reports* 13:2834.
- Morris C, Lecar H (1981). Voltage oscillations in the barnacle giant muscle fiber. *Biophys J* 35:193-213.
- Hodgkin AL, Huxley AF (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *J Physiol*.

## Author

Matt Tarchick  
Based on original retinal wave model from Tarchick et al. 2023

## License

MIT License
