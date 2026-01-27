# Visual Pathway Digital Twin - Architecture Diagram

```
                        VISUAL PATHWAY DIGITAL TWIN
                   Morris-Lecar Neurons with E/I/M Systems

    ╔══════════════════════════════════════════════════════════════════╗
    ║                           RETINA                                  ║
    ║  ┌─────────────────────┐         ┌─────────────────────┐         ║
    ║  │   BIPOLAR CELLS     │         │   GANGLION CELLS    │         ║
    ║  │   (Graded, glu)     │───E───▶│   (Spiking, glu)    │         ║
    ║  │   ON/OFF pathways   │         │   Projects to LGN   │         ║
    ║  └─────────────────────┘         └──────────┬──────────┘         ║
    ╚══════════════════════════════════════════════╪═══════════════════╝
                                                   │ E
                                                   ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     LATERAL GENICULATE (LGN)                      ║
    ║  ┌─────────────────────┐    I    ┌─────────────────────┐         ║
    ║  │   THALAMIC RELAY    │◀────────│  THALAMIC INHIBIT   │         ║
    ║  │   (Burst/tonic)     │         │  (Fast-spiking)     │         ║
    ║  │   T-type Ca²⁺       │────E───▶│  Gain control       │         ║
    ║  └──────────┬──────────┘         └─────────────────────┘         ║
    ╚═════════════╪════════════════════════════════════════════════════╝
                  │ E                    ▲
                  ▼                      │ E (feedback)
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    PRIMARY VISUAL CORTEX (V1)                     ║
    ║                                                                   ║
    ║  ┌─────────────────────┐    I    ┌─────────────────────┐         ║
    ║  │ CORTICAL PYRAMIDAL  │◀────────│ CORTICAL INHIBITORY │         ║
    ║  │ (Adapting, glu)     │         │ (PV+, fast-spiking) │         ║
    ║  │ Complex integration │────E───▶│ Perisomatic GABA    │         ║
    ║  └────────┬─┬──────────┘         └─────────▲───────────┘         ║
    ║           │ │ E                            │ I                    ║
    ║           │ │                              │                      ║
    ║           │ ▼  M                           │                      ║
    ║           │ ┌─────────────────────┐        │                      ║
    ║           └▶│ CORTICAL MODULATORY │────────┘                      ║
    ║             │ (SOM+/VIP+, slow)   │                               ║
    ║             │ Dendritic targeting │                               ║
    ║             │ GABA + neuropeptides│                               ║
    ║             └─────────────────────┘                               ║
    ╚══════════════════════════════════════════════════════════════════╝


                        SINGLE NEURON MODEL
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   MORRIS-LECAR BASE + HODGKIN-HUXLEY Na⁺ + E/I/M SYSTEMS         ║
    ║                                                                   ║
    ║   Voltage Equation:                                               ║
    ║   C_m dV/dt = I_leak + I_Ca + I_K + I_Na + I_TREK                ║
    ║             + I_MOD + I_E + I_I + I_noise + I_app                ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝


                      MODULATORY CASCADE (At → Bt → iMod)
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   Step 1: Modulatory NT → Metabotropic Receptor                  ║
    ║           ↓                                                       ║
    ║   Step 2: At (Logistic Growth)                                   ║
    ║           τ_A dA/dt = α_A · (Ca⁴ + H_M(M_mod)) · (1-A) - A       ║
    ║           ↓                                                       ║
    ║   Step 3: Bt (Second Messenger)                                  ║
    ║           τ_B dB/dt = β_B · A⁴ · (1-B) - B                       ║
    ║           ↓                                                       ║
    ║   Step 4: iMod (Ion Channel)                                     ║
    ║           I_MOD = -g_MOD · B · (V - E_MOD)                       ║
    ║                                                                   ║
    ║   E_MOD is TUNABLE:                                              ║
    ║   • E_MOD < E_rest → Inhibitory modulation                       ║
    ║   • E_MOD ≈ E_rest → Shunting modulation                         ║
    ║   • E_MOD > E_rest → Excitatory modulation                       ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝


                         SYNAPTIC CONNECTIVITY
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   • DISCRETE synaptic compartments (no PDEs for diffusion)       ║
    ║   • Neighboring neurons share NT values directly                 ║
    ║   • Sparse connectivity matrices for efficiency                  ║
    ║                                                                   ║
    ║   Connection Types:                                               ║
    ║   ┌─────────────┬────────────────────────────────────────────┐   ║
    ║   │ Excitatory  │ Glutamate, ACh → AMPA, nAChR (fast)        │   ║
    ║   │ Inhibitory  │ GABA → GABA-A (fast)                       │   ║
    ║   │ Modulatory  │ Peptides, mAChR → Metabotropic (slow)      │   ║
    ║   └─────────────┴────────────────────────────────────────────┘   ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝


                        PARAMETERS TO OPTIMIZE
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   Per Cell Type (~50 parameters each):                           ║
    ║                                                                   ║
    ║   Membrane:     C_m, g_leak, E_leak                              ║
    ║   Ionic:        g_Ca, g_K, g_Na, V1-V4 (gating)                  ║
    ║   Modulation:   g_TREK, g_MOD, E_MOD, τ_A, τ_B, α_A, β_B         ║
    ║   Synaptic:     g_E, g_I, g_M, τ_E, τ_I, τ_M                     ║
    ║   Release:      ρ_E, ρ_I, ρ_M, κ_E, κ_I, κ_M                     ║
    ║                                                                   ║
    ║   Optimization Strategy:                                          ║
    ║   1. Fit single-cell to patch-clamp data                         ║
    ║   2. Fit network to population recordings                        ║
    ║   3. Use genetic algorithms (as in Tarchick et al. 2023)         ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
```

## Key Equations Summary

### Morris-Lecar (voltage-gated channels)
- Ca²⁺ activation: `M_∞(V) = 0.5 * tanh((V - V₁)/V₂)`
- K⁺ activation: `N_∞(V) = 0.5 * tanh((V - V₃)/V₄)`
- K⁺ dynamics: `τ_N dN/dt = Λ(V) * (N_∞ - N)`

### Hodgkin-Huxley Na⁺
- `I_Na = -g_Na * M³ * H * (V - E_Na)`
- Standard α/β gating kinetics

### Neurotransmitter Release
- Sigmoidal: `Φ(V) = 1 / (1 + exp(-V_s * (V - V₀)))`
- Release: `τ_E dE/dt = ρ_E * Φ(V) - E`

### Receptor Activation
- Hill function: `H(NT) = NT² / (NT² + κ²)`

### Modulatory Cascade (from Tarchick et al. 2023)
- At: `τ_A dA/dt = α_A * Ca⁴ * (1-A) - A`
- Bt: `τ_B dB/dt = β_B * A⁴ * (1-B) - B`
- iMod: `I_MOD = -g_MOD * B * (V - E_MOD)`
