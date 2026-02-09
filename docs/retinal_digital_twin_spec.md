# Retinal Digital Twin — Technical Specification

## Implementation Target: Julia + DifferentialEquations.jl

**Author:** Generated for Dr. Matt Tarchick 
**Version:** 1.0  
**Date:** February 2026  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Cell Type Specifications](#3-cell-type-specifications)
4. [Synaptic Coupling Framework](#4-synaptic-coupling-framework)
5. [ERG Field Potential Calculation](#5-erg-field-potential-calculation)
6. [Stimulus Model](#6-stimulus-model)
7. [Code Architecture](#7-code-architecture)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Testing & Validation Strategy](#9-testing--validation-strategy)
10. [Parameter Tables](#10-parameter-tables)
11. [Literature References](#11-literature-references)
12. [Open Questions & Future Extensions](#12-open-questions--future-extensions)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The retinal digital twin simulates a **retinal column** — the minimal vertical pathway from photoreceptors to ganglion cells representing the tissue under a single ganglion cell receptive field. All neural dynamics (except the phototransduction cascade) use **Morris-Lecar (ML)** style equations with a unified neurotransmitter coupling framework using three modality channels: Excitatory (E), Inhibitory (I), and Modulatory (M).

### 1.2 Signal Flow

```
LIGHT
  │
  ▼
┌─────────────────────────────────┐
│  Photoreceptors (Rods × N_rod)  │──────────────────┐
│  Photoreceptors (Cones × N_cone)│──────────┐       │
└──────────┬──────────────────────┘          │       │
           │ Glutamate (graded)              │       │
           ▼                                 │       │
┌─────────────────────────┐                  │       │
│  Horizontal Cells × N_hc│◄────────────────┘       │
└──────────┬──────────────┘                          │
           │ Feedback to photoreceptors              │
           ▼                                         │
┌─────────────────┐  ┌──────────────────┐            │
│ ON-Bipolar × N_on│  │ OFF-Bipolar × N_off│          │
│ (mGluR6 inverting)│  │ (ionotropic)      │          │
└────────┬────────┘  └────────┬─────────┘            │
         │                    │                       │
         ▼                    ▼                       │
┌─────────────────────────────────────┐              │
│ A2 Amacrine × N_a2  (glycinergic)  │              │
│ GABAergic Amacrine × N_gaba        │              │
│ Dopaminergic Amacrine × N_dopa     │              │
└────────────────┬────────────────────┘              │
                 │                                    │
                 ▼                                    │
┌─────────────────────────┐                          │
│ Ganglion Cells × N_gc   │                          │
└─────────────────────────┘                          │
                                                      │
┌─────────────────────────┐                          │
│ Müller Glia × N_muller  │◄── K⁺ + Glu sensing ────┘
└─────────────────────────┘                          │
                                                      │
┌─────────────────────────┐                          │
│ RPE × N_rpe             │◄── photoreceptor K⁺ ────┘
└─────────────────────────┘
```

### 1.3 Population Sizes (Phase 1 Defaults)

These are tunable parameters. Default values represent a single retinal column:

| Cell Type | Symbol | Default N | Rationale |
|-----------|--------|-----------|-----------|
| Rods | N_rod | 20 | Convergence onto single bipolar |
| Cones | N_cone | 5 | Lower density, less convergence |
| Horizontal Cells | N_hc | 2 | Sparse lateral network |
| ON-Bipolar Cells | N_on | 1 | Single column |
| OFF-Bipolar Cells | N_off | 1 | Single column |
| A2 Amacrine | N_a2 | 3 | Multiple for OP dynamics |
| GABAergic Amacrine | N_gaba | 3 | Reciprocal with A2 |
| Dopaminergic Amacrine | N_dopa | 1 | Modulatory, sparse |
| Ganglion Cells | N_gc | 1 | Single column output |
| Müller Glia | N_muller | 1 | One per column |
| RPE | N_rpe | 1 | Sheet-like, one effective unit |

---

## 2. Mathematical Foundations

### 2.1 Morris-Lecar Template

All neural cell types use a common ML-style ODE system. The template is:

$$
C_m \frac{dV}{dt} = -g_L(V - E_L) - g_{Ca} \, m_\infty(V)(V - E_{Ca}) - g_K \, w \, (V - E_K) + I_{syn} + I_{ext} + I_{noise}
$$

$$
\frac{dw}{dt} = \phi \frac{w_\infty(V) - w}{\tau_w(V)}
$$

where the steady-state activation functions are:

$$
m_\infty(V) = \frac{1}{2}\left[1 + \tanh\left(\frac{V - V_1}{V_2}\right)\right]
$$

$$
w_\infty(V) = \frac{1}{2}\left[1 + \tanh\left(\frac{V - V_3}{V_4}\right)\right]
$$

$$
\tau_w(V) = \frac{1}{\cosh\left(\frac{V - V_3}{2 V_4}\right)}
$$

**Units convention throughout**: voltages in mV, currents in pA, conductances in nS, capacitance in pF, time in ms, concentrations in µM.

### 2.2 Synaptic Current Composition

The total synaptic current $I_{syn}$ for any postsynaptic neuron is:

$$
I_{syn} = I_E + I_I + I_M
$$

where each component is computed from presynaptic neurotransmitter concentrations:

$$
I_E = g_E \cdot s_E \cdot (V - E_E)
$$

$$
I_I = g_I \cdot s_I \cdot (V - E_I)
$$

$$
I_M = g_M \cdot f_M(s_M, V)
$$

Here $s_X$ is the normalized neurotransmitter concentration (0 to 1) at the postsynaptic site, $g_X$ is the maximal conductance, and $E_X$ is the reversal potential. The modulatory current $I_M$ has a more flexible form $f_M$ since modulation can alter conductances, time constants, or other parameters.

### 2.3 Neurotransmitter Release Dynamics

Presynaptic neurotransmitter release uses a sigmoid activation of voltage (or calcium where applicable):

$$
\frac{d[NT]}{dt} = \frac{\alpha_{NT} \cdot T_\infty(V_{pre}) - [NT]}{\tau_{NT}}
$$

where:

$$
T_\infty(V_{pre}) = \frac{1}{1 + \exp\left(\frac{-(V_{pre} - V_{th})}{V_{slope}}\right)}
$$

$\alpha_{NT}$ is the maximal release rate, $\tau_{NT}$ is the release/clearance time constant, $V_{th}$ is the half-activation voltage for release, and $V_{slope}$ controls steepness. This captures both graded release (photoreceptors, bipolars) and pulsatile release (ganglion cells) by adjusting parameters.

### 2.4 Postsynaptic Gating Variable

The normalized synaptic activation $s_X$ tracks neurotransmitter with its own dynamics:

$$
\frac{ds_X}{dt} = \frac{[NT]_{pre} - s_X}{\tau_{s,X}}
$$

This low-pass filters the presynaptic NT signal. For fast ionotropic synapses $\tau_{s,X} \approx 1$–$5$ ms; for metabotropic (mGluR6) $\tau_{s,X} \approx 10$–$50$ ms.

---

## 3. Cell Type Specifications

### 3.1 Photoreceptors (Rods and Cones)

Photoreceptors are unique: they use a **phototransduction cascade** for the light→current conversion, but the membrane potential dynamics follow a modified ML framework that includes the hyperpolarization-activated current $I_H$.

#### 3.1.1 Phototransduction Cascade (3-Compartment Model)

This is a simplified version of the Lamb & Pugh / Hamer cascade. Three state variables: activated rhodopsin/PDE activity ($R^*$), cGMP concentration ($G$), and intracellular calcium ($Ca$).

**Equation 1: Rhodopsin/PDE activation**

$$
\frac{dR^*}{dt} = \eta \cdot \Phi(t) - \frac{R^*}{\tau_R}
$$

- $\Phi(t)$: photon flux (photons/ms) — the stimulus
- $\eta$: quantum efficiency of photoisomerization ($\approx 0.67$ for rods, $\approx 0.5$ for cones)
- $\tau_R$: R* inactivation time constant (rods: $\approx 80$ ms, cones: $\approx 10$ ms)

**Equation 2: cGMP dynamics**

$$
\frac{dG}{dt} = \alpha_G \cdot \left(\frac{Ca_{dark}}{Ca}\right)^{n_{Ca}} - \beta_G \cdot (1 + \gamma \cdot R^*) \cdot G
$$

- $\alpha_G$: basal cGMP synthesis rate (modulated by Ca²⁺ feedback)
- $Ca_{dark}$: dark-adapted Ca²⁺ level
- $n_{Ca}$: cooperativity of Ca²⁺ feedback on guanylate cyclase ($\approx 2$–$4$)
- $\beta_G$: basal cGMP hydrolysis rate
- $\gamma$: gain of PDE activity on cGMP hydrolysis

The Ca²⁺ feedback term $\left(\frac{Ca_{dark}}{Ca}\right)^{n_{Ca}}$ accelerates cGMP synthesis when Ca drops (light adaptation).

**Equation 3: Calcium dynamics**

$$
\frac{dCa}{dt} = \frac{J_{CNG} \cdot f_{Ca} - k_{ex} \cdot Ca}{B_{Ca}}
$$

- $J_{CNG}$: current through CNG channels (proportional to $G^{n_G}$, see below)
- $f_{Ca}$: fraction of CNG current carried by Ca²⁺ ($\approx 0.12$)
- $k_{ex}$: Ca²⁺ extrusion rate via Na⁺/Ca²⁺-K⁺ exchanger
- $B_{Ca}$: effective Ca²⁺ buffering capacity

**CNG channel current (the photocurrent):**

$$
I_{CNG} = g_{CNG,max} \cdot \left(\frac{G}{G_{dark}}\right)^{n_G} \cdot (V - E_{CNG})
$$

- $n_G$: Hill coefficient for cGMP gating ($\approx 3$ for rods, $\approx 2.5$ for cones)
- $G_{dark}$: dark cGMP concentration (normalizes to dark current)
- $E_{CNG} \approx 0$ mV (non-selective cation channel)
- In darkness, $G = G_{dark}$, so $I_{CNG}$ is maximal (dark current)
- Light reduces $G$, reducing $I_{CNG}$, hyperpolarizing the cell

#### 3.1.2 Photoreceptor Membrane Potential

The membrane voltage follows an ML-like equation augmented with phototransduction and $I_H$:

$$
C_m \frac{dV}{dt} = -g_L(V - E_L) - I_{CNG} - I_H - I_{Kv} + I_{syn,feedback}
$$

**$I_H$: Hyperpolarization-activated current (critical for "nose" component)**

$$
\frac{dh}{dt} = \frac{h_\infty(V) - h}{\tau_H}
$$

$$
h_\infty(V) = \frac{1}{1 + \exp\left(\frac{V - V_{h,half}}{k_h}\right)}
$$

$$
I_H = g_H \cdot h \cdot (V - E_H)
$$

- $\tau_H \approx 50$ ms (tunable, key parameter for nose timing)
- $V_{h,half} \approx -70$ mV (half-activation voltage)
- $k_h \approx -10$ mV (slope, negative because activation increases with hyperpolarization)
- $E_H \approx -30$ mV (mixed Na⁺/K⁺ reversal)
- $g_H$: maximal I_H conductance (larger in rods than cones)

**$I_{Kv}$: Delayed rectifier (repolarizing)**

$$
I_{Kv} = g_{Kv} \cdot w_{Kv} \cdot (V - E_K)
$$

with standard ML gating for $w_{Kv}$.

**$I_{syn,feedback}$**: feedback from horizontal cells (Phase 1: can set to 0 or include as simple inhibitory conductance).

#### 3.1.3 Neurotransmitter Release (Glutamate)

Photoreceptors tonically release glutamate in the dark (depolarized state). Release is graded, proportional to a calcium-dependent process at the ribbon synapse:

$$
\frac{d[Glu]_{pre}}{dt} = \frac{\alpha_{Glu} \cdot R_{Glu}(V) - [Glu]_{pre}}{\tau_{Glu,rel}}
$$

$$
R_{Glu}(V) = \frac{1}{1 + \exp\left(\frac{-(V - V_{Glu,half})}{V_{Glu,slope}}\right)}
$$

- $V_{Glu,half} \approx -40$ mV
- $V_{Glu,slope} \approx 5$ mV
- In darkness ($V \approx -40$ mV): $R_{Glu} \approx 0.5$, tonic release
- In light ($V \approx -70$ mV): $R_{Glu} \approx 0$, release drops

#### 3.1.4 Rod vs. Cone Parameter Differences

| Parameter | Rod | Cone | Notes |
|-----------|-----|------|-------|
| $\tau_R$ | 80 ms | 10 ms | Faster cone R* inactivation |
| $n_G$ | 3.0 | 2.5 | CNG cooperativity |
| $g_{CNG,max}$ | 20 nS | 30 nS | Cones have larger dark current |
| $\tau_H$ | 50 ms | 30 ms | Faster cone I_H |
| $g_H$ | 2.0 nS | 1.0 nS | Less I_H in cones |
| $n_{Ca}$ | 4 | 2 | Stronger rod adaptation feedback |
| Sensitivity | ~1 photon | ~100 photons | Threshold for detectable response |
| $V_{dark}$ | -40 mV | -40 mV | Similar dark resting potential |

#### 3.1.5 State Variables per Photoreceptor

Each photoreceptor instance has 6 state variables:
- $R^*$ — activated rhodopsin/PDE
- $G$ — cGMP concentration
- $Ca$ — intracellular calcium
- $V$ — membrane potential
- $h$ — I_H gating variable
- $[Glu]_{pre}$ — released glutamate

Total for photoreceptor population: $6 \times (N_{rod} + N_{cone})$.

---

### 3.2 Horizontal Cells

Horizontal cells provide lateral inhibition and feedback to photoreceptors. They are GABAergic and modulate the photoreceptor calcium channel.

#### 3.2.1 Membrane Dynamics

Standard ML with one modification — a gap junction coupling term for lateral connectivity:

$$
C_m \frac{dV_{HC}}{dt} = -g_L(V_{HC} - E_L) - g_{Ca} \, m_\infty(V_{HC})(V_{HC} - E_{Ca}) - g_K \, w \, (V_{HC} - E_K) + I_{syn} + I_{gap}
$$

$$
I_{syn} = g_{Glu,HC} \cdot s_{Glu} \cdot (V_{HC} - E_E)
$$

where $s_{Glu}$ tracks the mean glutamate from all photoreceptors projecting to this HC.

**Gap junction coupling** (for Phase 2 spatial grid, but define the interface now):

$$
I_{gap} = \sum_{j \in neighbors} g_{gap}(V_j - V_{HC})
$$

In Phase 1 with $N_{hc} = 2$, this provides minimal coupling between the two HCs.

#### 3.2.2 HC Feedback to Photoreceptors

The HC feedback modulates the photoreceptor calcium channel (effectively shifts its voltage dependence). This is implemented as a modulatory signal:

$$
FB_{HC} = \bar{g}_{FB} \cdot \frac{1}{1 + \exp\left(\frac{-(V_{HC} - V_{FB,half})}{V_{FB,slope}}\right)}
$$

This feeds into the photoreceptor as a shift in the $I_{CNG}$ or as an additive current $I_{syn,feedback}$. Keep simple in Phase 1.

#### 3.2.3 State Variables per HC

- $V_{HC}$ — membrane potential
- $w_{HC}$ — ML recovery variable
- $s_{Glu,HC}$ — glutamate gating at HC

Total: $3 \times N_{hc}$.

---

### 3.3 ON-Bipolar Cells

The ON-bipolar cell is the dominant generator of the **b-wave**. It receives glutamate from photoreceptors via the **mGluR6 sign-inverting synapse**.

#### 3.3.1 mGluR6 Synapse (Sign Inversion)

Reference: Koz et al., "Modelling cholinergic retinal waves" (Scientific Reports).

The mGluR6 cascade: glutamate binding → Gαo activation → closure of TRPM1 channels. The key inversion is:

- **HIGH glutamate (dark)** → mGluR6 active → Gαo active → TRPM1 **CLOSED** → cell hyperpolarized
- **LOW glutamate (light)** → mGluR6 inactive → Gαo inactive → TRPM1 **OPEN** → cell depolarizes

Mathematically, the TRPM1 conductance is:

$$
g_{TRPM1}(t) = \bar{g}_{TRPM1} \cdot (1 - S_{mGluR6}(t))
$$

where $S_{mGluR6}$ tracks the mGluR6 cascade activation:

$$
\frac{dS_{mGluR6}}{dt} = \frac{\alpha_{mGluR6} \cdot [Glu]_{pre} - S_{mGluR6}}{\tau_{mGluR6}}
$$

- $\tau_{mGluR6} \approx 20$–$50$ ms (metabotropic, slower than ionotropic)
- $\alpha_{mGluR6}$: scaling factor
- When $[Glu]_{pre}$ is high: $S_{mGluR6} \rightarrow$ high, $g_{TRPM1} \rightarrow$ low (cell hyperpolarized)
- When $[Glu]_{pre}$ drops: $S_{mGluR6} \rightarrow$ low, $g_{TRPM1} \rightarrow$ high (cell depolarizes)

The TRPM1 current:

$$
I_{TRPM1} = g_{TRPM1} \cdot (V_{ON} - E_{TRPM1})
$$

where $E_{TRPM1} \approx 0$ mV (non-selective cation channel).

#### 3.3.2 Full ON-Bipolar Dynamics

$$
C_m \frac{dV_{ON}}{dt} = -g_L(V_{ON} - E_L) - g_{Ca} m_\infty(V_{ON})(V_{ON} - E_{Ca}) - g_K w_{ON}(V_{ON} - E_K) - I_{TRPM1} + I_{I,ON} + I_{M,ON}
$$

Note: $I_{TRPM1}$ is written with a negative sign because when $V_{ON} < E_{TRPM1}$, the current is inward (depolarizing), and we want the sign convention to produce depolarization when TRPM1 opens.

$I_{I,ON}$: inhibitory input from amacrine cells.
$I_{M,ON}$: modulatory input (dopamine).

#### 3.3.3 Glutamate Release from ON-Bipolar

ON-bipolars release glutamate in a graded fashion at their axon terminal:

$$
\frac{d[Glu]_{ON}}{dt} = \frac{\alpha_{Glu,ON} \cdot R_{Glu}(V_{ON}) - [Glu]_{ON}}{\tau_{Glu,ON}}
$$

Same sigmoid form as photoreceptors but with different half-activation (more depolarized, $V_{Glu,half} \approx -35$ mV).

#### 3.3.4 State Variables per ON-Bipolar

- $V_{ON}$ — membrane potential
- $w_{ON}$ — ML recovery variable
- $S_{mGluR6}$ — mGluR6 cascade state
- $[Glu]_{ON}$ — glutamate release

Total: $4 \times N_{on}$.

---

### 3.4 OFF-Bipolar Cells

OFF-bipolars use standard ionotropic (AMPA/KA) glutamate receptors. They depolarize when glutamate is high (in the dark) and hyperpolarize when glutamate drops (in the light). They are the primary generators of the **d-wave** at light offset.

#### 3.4.1 Ionotropic Synapse

$$
I_{Glu,OFF} = g_{iGluR} \cdot s_{Glu,OFF} \cdot (V_{OFF} - E_E)
$$

$$
\frac{ds_{Glu,OFF}}{dt} = \frac{[Glu]_{pre} - s_{Glu,OFF}}{\tau_{iGluR}}
$$

- $\tau_{iGluR} \approx 2$–$5$ ms (fast ionotropic)
- $E_E \approx 0$ mV

#### 3.4.2 Full OFF-Bipolar Dynamics

$$
C_m \frac{dV_{OFF}}{dt} = -g_L(V_{OFF} - E_L) - g_{Ca} m_\infty(V_{OFF})(V_{OFF} - E_{Ca}) - g_K w_{OFF}(V_{OFF} - E_K) + I_{Glu,OFF} + I_{I,OFF} + I_{M,OFF}
$$

#### 3.4.3 State Variables per OFF-Bipolar

- $V_{OFF}$, $w_{OFF}$, $s_{Glu,OFF}$, $[Glu]_{OFF}$ (its own glutamate release)

Total: $4 \times N_{off}$.

---

### 3.5 A2 (AII) Amacrine Cells — Glycinergic

A2 amacrines are narrow-field, fast-spiking interneurons critical for **oscillatory potential (OP) generation**. They receive excitatory input from ON-bipolars and provide glycinergic (inhibitory) output to OFF-bipolars and GABAergic amacrines.

#### 3.5.1 Dynamics

Fast ML parameters ($\phi$ large, $\tau_w$ small):

$$
C_m \frac{dV_{A2}}{dt} = -g_L(V_{A2} - E_L) - g_{Ca} m_\infty(V_{A2})(V_{A2} - E_{Ca}) - g_K w_{A2}(V_{A2} - E_K) + I_{E,A2} + I_{I,A2} + I_{M,A2}
$$

- $I_{E,A2}$: glutamate from ON-bipolars (ionotropic, fast)
- $I_{I,A2}$: GABAergic inhibition from GABAergic amacrines (reciprocal connection — this is the OP oscillator)
- $I_{M,A2}$: dopaminergic modulation

#### 3.5.2 Glycine Release

$$
\frac{d[Gly]}{dt} = \frac{\alpha_{Gly} \cdot T_\infty(V_{A2}) - [Gly]}{\tau_{Gly}}
$$

- $\tau_{Gly} \approx 3$–$5$ ms (fast, critical for OP frequency)

#### 3.5.3 State Variables per A2

- $V_{A2}$, $w_{A2}$, $[Gly]$ — 3 state variables

Total: $3 \times N_{a2}$.

---

### 3.6 GABAergic Amacrine Cells

Wide-field GABAergic amacrines form **reciprocal inhibitory connections** with A2 cells. The A2↔GABA reciprocal network is the primary oscillator driving OPs.

#### 3.6.1 Dynamics

Also fast ML parameters:

$$
C_m \frac{dV_{GABA}}{dt} = -g_L(V_{GABA} - E_L) - g_{Ca} m_\infty(V_{GABA})(V_{GABA} - E_{Ca}) - g_K w_{GABA}(V_{GABA} - E_K) + I_{E,GABA} + I_{I,GABA} + I_{M,GABA}
$$

- $I_{E,GABA}$: glutamate from ON-bipolars
- $I_{I,GABA}$: glycinergic inhibition from A2 cells (reciprocal)
- $I_{M,GABA}$: dopaminergic modulation

#### 3.6.2 GABA Release

$$
\frac{d[GABA]}{dt} = \frac{\alpha_{GABA} \cdot T_\infty(V_{GABA}) - [GABA]}{\tau_{GABA}}
$$

- $\tau_{GABA} \approx 5$–$10$ ms

#### 3.6.3 OP Generation Mechanism

The A2↔GABAergic reciprocal network generates OPs through mutual inhibition with different time constants:

1. ON-bipolar depolarization → excites both A2 and GABAergic amacrines
2. A2 fires first (slightly faster dynamics) → glycine inhibits GABAergic amacrine
3. GABAergic amacrine rebounds → GABA inhibits A2
4. A2 rebounds → cycle repeats

The oscillation frequency depends on:
- $\tau_{Gly}$ and $\tau_{GABA}$ (NT clearance rates)
- $\phi$ values of both cell types (intrinsic membrane speed)
- Coupling strengths $g_{Gly}$ and $g_{GABA}$

Target OP frequency: **100–160 Hz** (matching human ERG OPs).

#### 3.6.4 State Variables per GABAergic Amacrine

- $V_{GABA}$, $w_{GABA}$, $[GABA]$ — 3 state variables

Total: $3 \times N_{gaba}$.

---

### 3.7 Dopaminergic Amacrine Cells

Dopaminergic amacrines provide **modulatory** input that affects the gain and dynamics of other cell types. They modulate OP amplitude and contribute to light adaptation.

#### 3.7.1 Dynamics

Intermediate-speed ML:

$$
C_m \frac{dV_{DA}}{dt} = -g_L(V_{DA} - E_L) - g_{Ca} m_\infty(V_{DA})(V_{DA} - E_{Ca}) - g_K w_{DA}(V_{DA} - E_K) + I_{E,DA}
$$

- $I_{E,DA}$: glutamate from ON-bipolars (primary drive)

#### 3.7.2 Dopamine Release

$$
\frac{d[DA]}{dt} = \frac{\alpha_{DA} \cdot T_\infty(V_{DA}) - [DA]}{\tau_{DA}}
$$

- $\tau_{DA} \approx 100$–$500$ ms (slow modulatory release)

#### 3.7.3 Modulatory Effects

Dopamine modulates other cells via $I_M$ terms:
- **A2 amacrines**: Reduces coupling strength (via gap junction uncoupling in reality; modeled as reduced $g_{E,A2}$)
- **GABAergic amacrines**: Alters excitability
- **Horizontal cells**: Gap junction uncoupling (Phase 2)

Implementation: $I_M = g_M \cdot [DA] \cdot \Delta g_{target}$, where $\Delta g_{target}$ modifies a specific conductance of the target cell.

#### 3.7.4 State Variables per DA Amacrine

- $V_{DA}$, $w_{DA}$, $[DA]$ — 3 state variables

Total: $3 \times N_{dopa}$.

---

### 3.8 Ganglion Cells

Ganglion cells are the output neurons. They generate action potentials (important for future spike-based analyses but less critical for ERG waveform generation since ganglion cell contributions to the ERG field potential are relatively small compared to the mass potentials from bipolars and photoreceptors).

#### 3.8.1 Dynamics

Standard ML with parameters tuned for action potential generation:

$$
C_m \frac{dV_{GC}}{dt} = -g_L(V_{GC} - E_L) - g_{Ca} m_\infty(V_{GC})(V_{GC} - E_{Ca}) - g_K w_{GC}(V_{GC} - E_K) + I_{E,GC} + I_{I,GC}
$$

- $I_{E,GC}$: glutamate from ON-bipolars and OFF-bipolars
- $I_{I,GC}$: glycine from A2 amacrines, GABA from GABAergic amacrines

#### 3.8.2 State Variables per Ganglion Cell

- $V_{GC}$, $w_{GC}$ — 2 state variables

Total: $2 \times N_{gc}$.

---

### 3.9 Müller Glial Cells

Müller glia span the entire retinal thickness and generate the **slow P3 component** (slow negative wave after the a-wave) through K⁺ siphoning and glutamate uptake.

**Important note**: The c-wave is generated by RPE, NOT by Müller cells (per Dr. Koz's experimental findings).

#### 3.9.1 K⁺ Buffering Model

Müller cells sense extracellular K⁺ released by photoreceptors and bipolar cells:

$$
\frac{d[K^+]_o}{dt} = J_{K,neural} - J_{K,uptake} - J_{K,diffusion}
$$

$$
J_{K,neural} = \sum_{i \in \{PR, BC\}} \alpha_{K,i} \cdot I_{K,i}
$$

$$
J_{K,uptake} = g_{Kir} \cdot \frac{[K^+]_o}{[K^+]_o + K_{half}} \cdot (V_{Muller} - E_K([K^+]_o))
$$

The Müller cell generates a **current sink** at the inner retina (near ganglion cells) and a **current source** at the outer retina (near photoreceptors), producing the P3 field potential.

#### 3.9.2 Müller Cell Membrane Potential

Simplified model — primarily K⁺ permeable:

$$
C_{m,M} \frac{dV_M}{dt} = -g_{Kir,end} \cdot (V_M - E_K([K^+]_{o,end})) - g_{Kir,stalk} \cdot (V_M - E_K([K^+]_{o,stalk}))
$$

where the K⁺ Nernst potential updates dynamically:

$$
E_K([K^+]_o) = \frac{RT}{F} \ln\left(\frac{[K^+]_o}{[K^+]_i}\right)
$$

#### 3.9.3 Glutamate Sensing

Müller cells also sense extracellular glutamate via glutamate transporters:

$$
\frac{d[Glu]_o}{dt} = J_{Glu,release} - V_{max,EAAT} \cdot \frac{[Glu]_o}{K_m + [Glu]_o}
$$

This glutamate uptake generates an electrogenic current that contributes to the field potential.

#### 3.9.4 Müller Cell Contribution to ERG

The P3 component has:
- Onset: ~10–20 ms after flash (faster than RPE response)
- Time constant: ~200–500 ms
- Sign: **negative** in the ERG (same polarity as a-wave but slower)
- Sensitive to K⁺ channel blockers (e.g., BaCl₂)

#### 3.9.5 State Variables per Müller Cell

- $V_M$ — membrane potential
- $[K^+]_{o,end}$ — extracellular K⁺ at endfoot
- $[K^+]_{o,stalk}$ — extracellular K⁺ at stalk/outer region
- $[Glu]_o$ — extracellular glutamate

Total: $4 \times N_{muller}$.

---

### 3.10 RPE (Retinal Pigment Epithelium)

The RPE generates the **c-wave**, a slow positive component. This is driven by changes in subretinal K⁺ due to photoreceptor activity.

#### 3.10.1 Mechanism

Photoreceptor light response → reduced K⁺ efflux → drop in subretinal $[K^+]_o$ → RPE apical membrane hyperpolarizes → transepithelial potential change (c-wave).

$$
\frac{d[K^+]_{sub}}{dt} = J_{K,PR} - k_{RPE} \cdot ([K^+]_{sub} - [K^+]_{sub,rest})
$$

where $J_{K,PR}$ depends on the photoreceptor dark current (which includes K⁺ flux):

$$
J_{K,PR} = \alpha_{K,RPE} \cdot \sum_i I_{K,PR,i}
$$

#### 3.10.2 RPE Potential

$$
\frac{dV_{RPE}}{dt} = \frac{-g_{K,apical} \cdot (V_{RPE} - E_K([K^+]_{sub})) - g_{Cl,baso} \cdot (V_{RPE} - E_{Cl}) - g_L \cdot (V_{RPE} - E_L)}{\tau_{RPE}}
$$

- $\tau_{RPE}$: effective membrane time constant ($\approx 1000$–$5000$ ms — very slow)
- The c-wave peaks at $\sim 2$–$5$ seconds after flash

#### 3.10.3 State Variables per RPE

- $V_{RPE}$ — transepithelial potential
- $[K^+]_{sub}$ — subretinal potassium

Total: $2 \times N_{rpe}$.

---

## 4. Synaptic Coupling Framework

### 4.1 Connection Matrix

The retinal column connectivity is defined by a structured connection table. Each connection specifies: presynaptic cell type, postsynaptic cell type, neurotransmitter type (E/I/M), maximal conductance, reversal potential, and time constant.

| Pre → Post | NT Type | Receptor | $g_{max}$ (nS) | $E_{rev}$ (mV) | $\tau_s$ (ms) | Notes |
|------------|---------|----------|----------------|-----------------|---------------|-------|
| Rod → HC | E | iGluR | 5.0 | 0 | 3 | Fast ionotropic |
| Cone → HC | E | iGluR | 5.0 | 0 | 3 | Fast ionotropic |
| Rod → ON-BC | E | mGluR6 | — | — | 30 | Sign-inverting (see §3.3) |
| Cone → ON-BC | E | mGluR6 | — | — | 30 | Sign-inverting |
| Rod → OFF-BC | E | iGluR | 4.0 | 0 | 3 | Standard ionotropic |
| Cone → OFF-BC | E | iGluR | 4.0 | 0 | 3 | Standard ionotropic |
| HC → PR | M | feedback | 1.0 | — | 20 | Ca²⁺ channel modulation |
| ON-BC → A2 | E | iGluR | 8.0 | 0 | 2 | Fast, drives OPs |
| ON-BC → GABA-AC | E | iGluR | 6.0 | 0 | 2 | Fast |
| ON-BC → DA-AC | E | iGluR | 3.0 | 0 | 5 | Moderate |
| ON-BC → GC | E | iGluR | 5.0 | 0 | 3 | Direct pathway |
| OFF-BC → GC | E | iGluR | 5.0 | 0 | 3 | Direct pathway |
| A2 → GABA-AC | I | GlyR | 10.0 | -80 | 4 | Reciprocal (OP oscillator) |
| A2 → OFF-BC | I | GlyR | 5.0 | -80 | 4 | Cross-pathway inhibition |
| A2 → GC | I | GlyR | 3.0 | -80 | 4 | |
| GABA-AC → A2 | I | GABA_A | 10.0 | -70 | 8 | Reciprocal (OP oscillator) |
| GABA-AC → ON-BC | I | GABA_A | 3.0 | -70 | 8 | Feedback inhibition |
| GABA-AC → GC | I | GABA_A | 3.0 | -70 | 8 | |
| DA-AC → A2 | M | D1R | 1.0 | — | 200 | Gain modulation |
| DA-AC → GABA-AC | M | D1R | 1.0 | — | 200 | Gain modulation |
| DA-AC → HC | M | D1R | 0.5 | — | 200 | Gap junction uncoupling |

### 4.2 Population Averaging

When multiple presynaptic cells of the same type project to a single postsynaptic cell, their neurotransmitter concentrations are **averaged**:

$$
[NT]_{eff} = \frac{1}{N_{pre}} \sum_{i=1}^{N_{pre}} [NT]_i
$$

This is biologically reasonable for the retinal column where all photoreceptors in the column project to the same bipolar cell.

### 4.3 Modulatory Synapse Implementation

For modulatory (M-type) synapses, the effect is a multiplicative gain change on a target conductance rather than a direct current:

$$
g_{target}' = g_{target} \cdot (1 + \kappa_M \cdot [DA])
$$

where $\kappa_M$ can be positive (facilitation) or negative (suppression) depending on the specific modulatory effect.

---

## 5. ERG Field Potential Calculation

### 5.1 Principle

The ERG is a **field potential** recorded at the cornea. It represents the weighted sum of all transmembrane currents in the retina. The weight of each cell type's contribution depends on:

1. **Current dipole strength** — how much current flows across the cell
2. **Cell geometry** — oriented current sources/sinks produce stronger extracellular fields
3. **Population size** — more cells = larger aggregate signal
4. **Distance to electrode** — further = weaker (but retina is thin, so this is a modest factor)

### 5.2 ERG Calculation Formula

$$
V_{ERG}(t) = \sum_{k \in \text{cell types}} w_k \cdot N_k \cdot I_{trans,k}(t)
$$

where $I_{trans,k}$ is the total transmembrane current of a single cell of type $k$:

$$
I_{trans,k} = C_m \frac{dV_k}{dt}
$$

or equivalently, the sum of all ionic and synaptic currents (with appropriate sign).

### 5.3 ERG Weights

The weights $w_k$ capture cell geometry and electrode distance effects. These are key fitting parameters:

| Cell Type | Weight $w_k$ | Sign Convention | Dominant ERG Component |
|-----------|-------------|-----------------|----------------------|
| Rod | $w_{rod}$ = 1.0 | Hyperpolarization → negative | a-wave (negative) |
| Cone | $w_{cone}$ = 0.5 | Same as rod | a-wave (photopic) |
| ON-Bipolar | $w_{on}$ = -2.0 | Depolarization → positive (note negative weight × depolarizing current) | b-wave (positive) |
| OFF-Bipolar | $w_{off}$ = -1.0 | Depolarization at light-off → positive | d-wave |
| A2 Amacrine | $w_{a2}$ = 0.3 | Fast oscillations | OPs |
| GABA Amacrine | $w_{gaba}$ = 0.3 | Fast oscillations | OPs |
| DA Amacrine | $w_{da}$ = 0.05 | Minimal direct contribution | — |
| Ganglion Cell | $w_{gc}$ = 0.1 | Spikes (small contribution) | PhNR |
| Müller Glia | $w_{muller}$ = -1.5 | K⁺ siphoning → negative | P3 (slow negative) |
| RPE | $w_{rpe}$ = -1.0 | Apical hyperpolarization → positive at cornea | c-wave |

**Critical note**: The sign convention requires careful attention. The b-wave is positive in the ERG despite being generated by a depolarizing current. This is because the ON-bipolar current dipole (current flowing from inner to outer retina) produces a positive potential at the corneal surface. The weight incorporates this geometric sign inversion.

### 5.4 Decomposition Outputs

For analysis and debugging, output the ERG decomposed by component:

$$
V_{ERG,k}(t) = w_k \cdot N_k \cdot I_{trans,k}(t) \quad \forall k
$$

Also compute the "classic" ERG components:
- **a-wave**: $V_{ERG,rod} + V_{ERG,cone}$
- **b-wave**: $V_{ERG,on}$
- **P3**: $V_{ERG,muller}$
- **OPs**: bandpass filter (75–300 Hz) of $V_{ERG,a2} + V_{ERG,gaba}$
- **d-wave**: $V_{ERG,off}$
- **c-wave**: $V_{ERG,rpe}$

---

## 6. Stimulus Model

### 6.1 Light Stimulus

The stimulus is specified as photon flux $\Phi(t)$ in photons·µm⁻²·ms⁻¹ (or equivalently, a scotopic/photopic intensity in log units relative to threshold).

#### 6.1.1 Single Flash

$$
\Phi(t) = \begin{cases} I_0 & \text{if } t_{on} \leq t \leq t_{on} + t_{dur} \\ 0 & \text{otherwise} \end{cases}
$$

Parameters:
- $I_0$: flash intensity (variable, log scale spanning ~6 orders of magnitude)
- $t_{on}$: flash onset time (default: 200 ms, to allow baseline)
- $t_{dur}$: flash duration (default: 10 ms for brief flash; variable for step stimuli)

#### 6.1.2 Intensity Scale

Define a log intensity scale:

| Log Intensity | Regime | Dominant Pathway |
|---------------|--------|-----------------|
| -3 to -1 | Scotopic | Rod-driven |
| -1 to 1 | Mesopic | Mixed rod/cone |
| 1 to 3 | Photopic | Cone-driven |

Map to photon flux: $I_0 = I_{ref} \cdot 10^{log\_intensity}$, where $I_{ref}$ is calibrated so that $\log = 0$ is approximately the rod-cone transition.

#### 6.1.3 Rod vs. Cone Sensitivity

Rods and cones differ in quantum catch efficiency and gain:
- Rods: sensitive to ~1 photon (single photon response ~1 pA)
- Cones: require ~100 photons for equivalent response
- Implement by scaling $\eta$ (quantum efficiency) and $\gamma$ (PDE gain) differently

#### 6.1.4 Future Stimulus Types

Define the interface to support (but don't implement yet):
- Flash trains (periodic flashes)
- Flicker (sinusoidal modulation)
- Step stimuli (sustained illumination)
- Paired flash (for recovery kinetics)

---

## 7. Code Architecture

### 7.1 Module Structure

```
RetinalTwin/
├── src/
│   ├── RetinalTwin.jl           # Main module, exports
│   ├── types.jl                  # All struct definitions
│   ├── parameters.jl             # Default parameter sets
│   ├── cells/
│   │   ├── photoreceptor.jl      # Phototransduction + membrane
│   │   ├── horizontal.jl         # HC dynamics
│   │   ├── on_bipolar.jl         # mGluR6 + ML dynamics
│   │   ├── off_bipolar.jl        # Ionotropic + ML dynamics
│   │   ├── a2_amacrine.jl        # Glycinergic fast dynamics
│   │   ├── gaba_amacrine.jl      # GABAergic fast dynamics
│   │   ├── da_amacrine.jl        # Dopaminergic modulation
│   │   ├── ganglion.jl           # GC dynamics
│   │   ├── muller.jl             # Müller glia K+ buffering
│   │   └── rpe.jl                # RPE c-wave
│   ├── synapses/
│   │   ├── synapse.jl            # Synaptic transmission framework
│   │   ├── mglur6.jl             # Sign-inverting synapse
│   │   └── modulatory.jl         # Dopaminergic modulation
│   ├── circuit/
│   │   ├── retinal_column.jl     # Column assembly + wiring
│   │   └── connectivity.jl       # Connection matrix definition
│   ├── stimulus/
│   │   └── light.jl              # Stimulus protocols
│   ├── erg/
│   │   └── field_potential.jl    # ERG calculation + decomposition
│   ├── simulation/
│   │   ├── ode_system.jl         # Assemble full ODE system
│   │   └── run.jl                # Simulation driver
│   └── visualization/
│       └── plots.jl              # Plotting utilities
├── test/
│   ├── runtests.jl
│   ├── test_photoreceptor.jl
│   ├── test_mglur6.jl
│   ├── test_oscillatory.jl
│   └── test_erg.jl
└── Project.toml
```

### 7.2 Core Type Definitions

```julia
# ============================================================
# types.jl — Core type definitions
# ============================================================

using StaticArrays
using Parameters

"""
    NeurotransmitterState

Tracks E/I/M neurotransmitter concentrations for a single cell.
"""
@with_kw mutable struct NeurotransmitterState
    E::Float64 = 0.0   # Excitatory (glutamate)
    I::Float64 = 0.0   # Inhibitory (GABA or glycine)
    M::Float64 = 0.0   # Modulatory (dopamine)
end

"""
    MLParams

Morris-Lecar parameters for a single cell type.
"""
@with_kw struct MLParams
    C_m::Float64 = 20.0     # pF - membrane capacitance
    g_L::Float64 = 2.0      # nS - leak conductance
    g_Ca::Float64 = 4.0     # nS - calcium conductance
    g_K::Float64 = 8.0      # nS - potassium conductance
    E_L::Float64 = -60.0    # mV - leak reversal
    E_Ca::Float64 = 120.0   # mV - calcium reversal
    E_K::Float64 = -84.0    # mV - potassium reversal
    V1::Float64 = -1.2      # mV - m_inf half-activation
    V2::Float64 = 18.0      # mV - m_inf slope
    V3::Float64 = 2.0       # mV - w_inf half-activation
    V4::Float64 = 30.0      # mV - w_inf slope
    phi::Float64 = 0.04     # dimensionless - w time constant scaling
end

"""
    PhototransductionParams

Parameters for the 3-compartment phototransduction cascade.
"""
@with_kw struct PhototransductionParams
    eta::Float64 = 0.67         # quantum efficiency
    tau_R::Float64 = 80.0       # ms - R* inactivation
    alpha_G::Float64 = 20.0     # µM/ms - basal cGMP synthesis
    beta_G::Float64 = 0.5       # 1/ms - basal cGMP hydrolysis
    gamma_PDE::Float64 = 5.0    # gain of R* on PDE
    n_Ca::Int = 4               # Ca feedback cooperativity
    Ca_dark::Float64 = 0.3      # µM - dark calcium
    G_dark::Float64 = 5.0       # µM - dark cGMP
    n_G::Float64 = 3.0          # Hill coeff for CNG
    g_CNG_max::Float64 = 20.0   # nS - max CNG conductance
    E_CNG::Float64 = 0.0        # mV - CNG reversal
    f_Ca::Float64 = 0.12        # fraction of CNG current carried by Ca
    k_ex::Float64 = 0.1         # 1/ms - Ca extrusion rate
    B_Ca::Float64 = 50.0        # Ca buffering capacity
    # iH channel parameters
    g_H::Float64 = 2.0          # nS - max I_H conductance
    tau_H::Float64 = 50.0       # ms - I_H time constant (TUNABLE)
    V_h_half::Float64 = -70.0   # mV - I_H half-activation
    k_h::Float64 = -10.0        # mV - I_H slope
    E_H::Float64 = -30.0        # mV - I_H reversal (mixed Na/K)
    # Voltage-gated K
    g_Kv::Float64 = 3.0         # nS
    # Glutamate release
    alpha_Glu::Float64 = 1.0    # max glutamate
    V_Glu_half::Float64 = -40.0 # mV
    V_Glu_slope::Float64 = 5.0  # mV
    tau_Glu::Float64 = 5.0      # ms - release time constant
end

"""
    SynapseParams

Parameters for a single synaptic connection type.
"""
@with_kw struct SynapseParams
    g_max::Float64              # nS - maximal conductance
    E_rev::Float64              # mV - reversal potential (NaN for modulatory)
    tau_s::Float64              # ms - postsynaptic gating time constant
    nt_type::Symbol = :E        # :E, :I, or :M
    receptor::Symbol = :iGluR   # :iGluR, :mGluR6, :GlyR, :GABA_A, :D1R, :feedback
end

"""
    mGluR6Params

Parameters specific to the mGluR6 sign-inverting synapse.
"""
@with_kw struct mGluR6Params
    g_TRPM1_max::Float64 = 10.0  # nS
    E_TRPM1::Float64 = 0.0       # mV (non-selective cation)
    alpha_mGluR6::Float64 = 1.0   # scaling
    tau_mGluR6::Float64 = 30.0    # ms - metabotropic time constant
end

"""
    MullerParams

Müller glial cell parameters.
"""
@with_kw struct MullerParams
    C_m::Float64 = 30.0          # pF
    g_Kir_end::Float64 = 5.0     # nS - endfoot Kir conductance
    g_Kir_stalk::Float64 = 2.0   # nS - stalk Kir conductance
    K_o_rest::Float64 = 3.0      # mM - resting extracellular K+
    K_i::Float64 = 140.0         # mM - intracellular K+
    tau_K_diffusion::Float64 = 200.0 # ms - K+ clearance
    alpha_K::Float64 = 0.001     # K+ release per unit current
    # Glutamate transport
    V_max_EAAT::Float64 = 0.5    # µM/ms
    K_m_EAAT::Float64 = 10.0     # µM
end

"""
    RPEParams

RPE parameters for c-wave generation.
"""
@with_kw struct RPEParams
    tau_RPE::Float64 = 3000.0    # ms - very slow dynamics
    g_K_apical::Float64 = 5.0    # nS
    g_Cl_baso::Float64 = 2.0     # nS
    g_L_RPE::Float64 = 0.5       # nS
    E_Cl::Float64 = -50.0        # mV
    E_L_RPE::Float64 = -60.0     # mV
    K_sub_rest::Float64 = 3.0    # mM
    k_RPE::Float64 = 0.0005     # K+ clearance rate
    alpha_K_RPE::Float64 = 0.001 # K+ flux scaling
end

"""
    ERGWeights

Weights for computing the ERG field potential from cell currents.
"""
@with_kw struct ERGWeights
    rod::Float64 = 1.0
    cone::Float64 = 0.5
    on_bc::Float64 = -2.0
    off_bc::Float64 = -1.0
    a2::Float64 = 0.3
    gaba::Float64 = 0.3
    da::Float64 = 0.05
    gc::Float64 = 0.1
    muller::Float64 = -1.5
    rpe::Float64 = -1.0
end

"""
    StimulusProtocol

Light stimulus specification.
"""
@with_kw struct StimulusProtocol
    I_0::Float64 = 1000.0       # photons/µm²/ms
    t_on::Float64 = 200.0       # ms - flash onset
    t_dur::Float64 = 10.0       # ms - flash duration
    background::Float64 = 0.0   # background illumination
end

"""
    RetinalColumn

Complete state and parameters for one retinal column.
"""
struct RetinalColumn
    # Population sizes
    n_rod::Int
    n_cone::Int
    n_hc::Int
    n_on::Int
    n_off::Int
    n_a2::Int
    n_gaba::Int
    n_dopa::Int
    n_gc::Int
    n_muller::Int
    n_rpe::Int
    
    # Parameter sets (one per cell type)
    rod_params::PhototransductionParams
    cone_params::PhototransductionParams
    hc_params::MLParams
    on_params::MLParams
    off_params::MLParams
    a2_params::MLParams
    gaba_params::MLParams
    da_params::MLParams
    gc_params::MLParams
    muller_params::MullerParams
    rpe_params::RPEParams
    
    # Synapse parameters
    mglur6_params::mGluR6Params
    
    # ERG weights
    erg_weights::ERGWeights
    
    # Stimulus
    stimulus::StimulusProtocol
end
```

### 7.3 State Vector Layout

The ODE system uses a flat state vector `u` with named index ranges for each cell population. Total state variables:

| Cell Type | Vars/Cell | × Population | Total |
|-----------|-----------|-------------|-------|
| Rod | 6 | 20 | 120 |
| Cone | 6 | 5 | 30 |
| HC | 3 | 2 | 6 |
| ON-BC | 4 | 1 | 4 |
| OFF-BC | 4 | 1 | 4 |
| A2 | 3 | 3 | 9 |
| GABA-AC | 3 | 3 | 9 |
| DA-AC | 3 | 1 | 3 |
| GC | 2 | 1 | 2 |
| Müller | 4 | 1 | 4 |
| RPE | 2 | 1 | 2 |
| **Total** | | | **193** |

This is very tractable — 193 ODEs is trivial for modern solvers.

```julia
"""
    StateIndex

Named indices into the flat state vector.
Computed from population sizes at construction time.
"""
struct StateIndex
    # For each cell type: UnitRange{Int} into u vector
    # Photoreceptors: [R*, G, Ca, V, h, Glu] × N
    rod::UnitRange{Int}
    cone::UnitRange{Int}
    # HC: [V, w, s_Glu] × N
    hc::UnitRange{Int}
    # ON-BC: [V, w, S_mGluR6, Glu] × N
    on_bc::UnitRange{Int}
    # OFF-BC: [V, w, s_Glu, Glu_release] × N
    off_bc::UnitRange{Int}
    # A2: [V, w, Gly] × N
    a2::UnitRange{Int}
    # GABA-AC: [V, w, GABA] × N
    gaba_ac::UnitRange{Int}
    # DA-AC: [V, w, DA] × N
    da_ac::UnitRange{Int}
    # GC: [V, w] × N
    gc::UnitRange{Int}
    # Müller: [V_M, K_o_end, K_o_stalk, Glu_o] × N
    muller::UnitRange{Int}
    # RPE: [V_RPE, K_sub] × N
    rpe::UnitRange{Int}
    # Total length
    total::Int
end

function StateIndex(col::RetinalColumn)
    idx = 1
    function next_range(n_cells, vars_per_cell)
        r = idx:(idx + n_cells * vars_per_cell - 1)
        idx += n_cells * vars_per_cell
        return r
    end
    
    rod = next_range(col.n_rod, 6)
    cone = next_range(col.n_cone, 6)
    hc = next_range(col.n_hc, 3)
    on_bc = next_range(col.n_on, 4)
    off_bc = next_range(col.n_off, 4)
    a2 = next_range(col.n_a2, 3)
    gaba_ac = next_range(col.n_gaba, 3)
    da_ac = next_range(col.n_dopa, 3)
    gc = next_range(col.n_gc, 2)
    muller = next_range(col.n_muller, 4)
    rpe = next_range(col.n_rpe, 2)
    
    return StateIndex(rod, cone, hc, on_bc, off_bc, a2, gaba_ac, da_ac, gc, muller, rpe, idx - 1)
end
```

### 7.4 ODE Right-Hand-Side Function

```julia
"""
    retinal_column_rhs!(du, u, p, t)

In-place ODE right-hand side for the full retinal column.
`p` contains the RetinalColumn and StateIndex.

This function:
1. Unpacks state variables
2. Computes stimulus
3. Updates each cell type's derivatives
4. Handles synaptic coupling
"""
function retinal_column_rhs!(du, u, p, t)
    col, idx = p
    
    # --- Stimulus ---
    Phi = compute_stimulus(col.stimulus, t)
    
    # --- Phase 1: Compute neurotransmitter concentrations ---
    # (read current NT levels from state vector)
    glu_rod_mean = mean_nt(u, idx.rod, 6, 6, col.n_rod)     # Glu is var 6 in PR
    glu_cone_mean = mean_nt(u, idx.cone, 6, 6, col.n_cone)
    glu_pr_mean = weighted_mean(glu_rod_mean, col.n_rod, 
                                glu_cone_mean, col.n_cone)
    
    # ON-bipolar glutamate
    glu_on_mean = mean_nt(u, idx.on_bc, 4, 4, col.n_on)
    
    # OFF-bipolar glutamate  
    glu_off_mean = mean_nt(u, idx.off_bc, 4, 4, col.n_off)
    
    # Amacrine NTs
    gly_a2_mean = mean_nt(u, idx.a2, 3, 3, col.n_a2)
    gaba_mean = mean_nt(u, idx.gaba_ac, 3, 3, col.n_gaba)
    da_mean = mean_nt(u, idx.da_ac, 3, 3, col.n_dopa)
    
    # --- Phase 2: Update each cell population ---
    
    # Photoreceptors
    for i in 1:col.n_rod
        offset = idx.rod[1] + (i-1) * 6
        update_photoreceptor!(view(du, offset:offset+5), 
                              view(u, offset:offset+5),
                              col.rod_params, Phi, 0.0)  # HC feedback = 0 for now
    end
    
    for i in 1:col.n_cone
        offset = idx.cone[1] + (i-1) * 6
        update_photoreceptor!(view(du, offset:offset+5),
                              view(u, offset:offset+5),
                              col.cone_params, Phi, 0.0)
    end
    
    # Horizontal cells
    for i in 1:col.n_hc
        offset = idx.hc[1] + (i-1) * 3
        update_horizontal!(view(du, offset:offset+2),
                          view(u, offset:offset+2),
                          col.hc_params, glu_pr_mean)
    end
    
    # ON-Bipolar (mGluR6)
    for i in 1:col.n_on
        offset = idx.on_bc[1] + (i-1) * 4
        update_on_bipolar!(view(du, offset:offset+3),
                          view(u, offset:offset+3),
                          col.on_params, col.mglur6_params,
                          glu_pr_mean, gaba_mean, da_mean)
    end
    
    # OFF-Bipolar
    for i in 1:col.n_off
        offset = idx.off_bc[1] + (i-1) * 4
        update_off_bipolar!(view(du, offset:offset+3),
                           view(u, offset:offset+3),
                           col.off_params,
                           glu_pr_mean, gaba_mean, da_mean)
    end
    
    # A2 Amacrine
    for i in 1:col.n_a2
        offset = idx.a2[1] + (i-1) * 3
        update_a2!(view(du, offset:offset+2),
                  view(u, offset:offset+2),
                  col.a2_params,
                  glu_on_mean, gaba_mean, da_mean)
    end
    
    # GABAergic Amacrine
    for i in 1:col.n_gaba
        offset = idx.gaba_ac[1] + (i-1) * 3
        update_gaba_amacrine!(view(du, offset:offset+2),
                             view(u, offset:offset+2),
                             col.gaba_params,
                             glu_on_mean, gly_a2_mean, da_mean)
    end
    
    # DA Amacrine
    for i in 1:col.n_dopa
        offset = idx.da_ac[1] + (i-1) * 3
        update_da_amacrine!(view(du, offset:offset+2),
                           view(u, offset:offset+2),
                           col.da_params, glu_on_mean)
    end
    
    # Ganglion cells
    for i in 1:col.n_gc
        offset = idx.gc[1] + (i-1) * 2
        update_ganglion!(view(du, offset:offset+1),
                        view(u, offset:offset+1),
                        col.gc_params,
                        glu_on_mean, glu_off_mean,
                        gly_a2_mean, gaba_mean)
    end
    
    # Müller glia
    for i in 1:col.n_muller
        offset = idx.muller[1] + (i-1) * 4
        # Need total K+ currents from PRs and bipolars
        update_muller!(view(du, offset:offset+3),
                      view(u, offset:offset+3),
                      col.muller_params,
                      u, idx, col)  # Pass full state for K+ sensing
    end
    
    # RPE
    for i in 1:col.n_rpe
        offset = idx.rpe[1] + (i-1) * 2
        update_rpe!(view(du, offset:offset+1),
                   view(u, offset:offset+1),
                   col.rpe_params,
                   u, idx, col)  # Pass full state for K+ sensing
    end
    
    return nothing
end
```

### 7.5 Key Cell Update Functions (Signatures)

```julia
"""
    update_photoreceptor!(du, u, params, Phi, I_feedback)

Update derivatives for one photoreceptor.
u = [R*, G, Ca, V, h, Glu]
"""
function update_photoreceptor!(du, u, params::PhototransductionParams, 
                                Phi::Float64, I_feedback::Float64)
    R_star, G, Ca, V, h, Glu = u[1], u[2], u[3], u[4], u[5], u[6]
    
    # Phototransduction
    du[1] = params.eta * Phi - R_star / params.tau_R
    
    # cGMP
    Ca_ratio = (params.Ca_dark / max(Ca, 1e-6))^params.n_Ca
    du[2] = params.alpha_G * Ca_ratio - params.beta_G * (1.0 + params.gamma_PDE * R_star) * G
    
    # CNG current
    G_norm = (G / params.G_dark)^params.n_G
    I_CNG = params.g_CNG_max * G_norm * (V - params.E_CNG)
    
    # Calcium
    du[3] = (I_CNG * params.f_Ca - params.k_ex * Ca) / params.B_Ca
    
    # I_H
    h_inf = 1.0 / (1.0 + exp((V - params.V_h_half) / params.k_h))
    I_H = params.g_H * h * (V - params.E_H)
    du[5] = (h_inf - h) / params.tau_H
    
    # I_Kv (simplified)
    w_Kv = 1.0 / (1.0 + exp(-(V + 30.0) / 10.0))  # simplified steady-state
    I_Kv = params.g_Kv * w_Kv * (V - (-84.0))
    
    # Membrane potential
    g_L = 2.0  # nS
    E_L = -70.0  # mV (photoreceptor dark resting ~ -40 mV set by dark current)
    C_m = 20.0  # pF
    du[4] = (-g_L * (V - E_L) - I_CNG - I_H - I_Kv + I_feedback) / C_m
    
    # Glutamate release
    R_glu = 1.0 / (1.0 + exp(-(V - params.V_Glu_half) / params.V_Glu_slope))
    du[6] = (params.alpha_Glu * R_glu - Glu) / params.tau_Glu
    
    return nothing
end

"""
    update_on_bipolar!(du, u, ml_params, mglur6_params, glu_pre, gaba, da)

Update derivatives for one ON-bipolar cell.
u = [V, w, S_mGluR6, Glu_release]
"""
function update_on_bipolar!(du, u, ml::MLParams, mg::mGluR6Params,
                            glu_pre::Float64, gaba::Float64, da::Float64)
    V, w, S, Glu_rel = u[1], u[2], u[3], u[4]
    
    # mGluR6 cascade
    du[3] = (mg.alpha_mGluR6 * glu_pre - S) / mg.tau_mGluR6
    
    # TRPM1 current (sign-inverted: low S → high conductance)
    g_TRPM1 = mg.g_TRPM1_max * (1.0 - S)
    I_TRPM1 = g_TRPM1 * (V - mg.E_TRPM1)
    
    # ML dynamics
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))
    
    # Inhibitory current (GABAergic)
    I_I = 3.0 * gaba * (V - (-70.0))  # g_GABA * s_GABA * (V - E_GABA)
    
    # Modulatory (dopamine gain adjustment)
    g_mod = 1.0 + 0.5 * da  # simple gain factor
    
    du[1] = (-ml.g_L * (V - ml.E_L) - 
              ml.g_Ca * m_inf * (V - ml.E_Ca) - 
              ml.g_K * w * (V - ml.E_K) - 
              I_TRPM1 * g_mod + I_I) / ml.C_m
    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)
    
    # Glutamate release
    R_glu = 1.0 / (1.0 + exp(-(V - (-35.0)) / 5.0))
    du[4] = (R_glu - Glu_rel) / 5.0
    
    return nothing
end
```

### 7.6 Simulation Driver

```julia
"""
    simulate_flash(; intensity=1000.0, duration=10.0, 
                    t_total=5000.0, dt_save=0.1,
                    regime=:scotopic, kwargs...)

Main simulation entry point. Returns a NamedTuple with:
- t: time vector
- erg: ERG trace
- erg_components: Dict of per-cell-type ERG contributions
- cell_voltages: Dict of voltage traces by cell type
"""
function simulate_flash(; intensity=1000.0, duration=10.0,
                         t_total=5000.0, dt_save=0.1,
                         regime=:scotopic, kwargs...)
    
    # Build retinal column with default or custom parameters
    col = build_retinal_column(; regime=regime, kwargs...)
    col = @set col.stimulus.I_0 = intensity
    col = @set col.stimulus.t_dur = duration
    
    # Build state index
    sidx = StateIndex(col)
    
    # Initial conditions (dark-adapted)
    u0 = dark_adapted_state(col, sidx)
    
    # ODE problem
    tspan = (0.0, t_total)
    p = (col, sidx)
    prob = ODEProblem(retinal_column_rhs!, u0, tspan, p)
    
    # Solve
    sol = solve(prob, Tsit5();  # or TRBDF2 for stiff systems
                saveat=dt_save,
                abstol=1e-8, reltol=1e-6,
                maxiters=1_000_000)
    
    # Compute ERG
    erg, components = compute_erg(sol, col, sidx)
    
    # Extract voltages
    voltages = extract_voltages(sol, col, sidx)
    
    return (t=sol.t, erg=erg, erg_components=components, 
            cell_voltages=voltages, solution=sol)
end

"""
    dark_adapted_state(col, sidx)

Compute initial conditions for dark-adapted retina.
"""
function dark_adapted_state(col::RetinalColumn, sidx::StateIndex)
    u0 = zeros(sidx.total)
    
    # Photoreceptors: dark state
    for i in 1:col.n_rod
        offset = sidx.rod[1] + (i-1) * 6
        u0[offset]   = 0.0              # R* = 0 (no light)
        u0[offset+1] = col.rod_params.G_dark  # G = G_dark
        u0[offset+2] = col.rod_params.Ca_dark  # Ca = Ca_dark
        u0[offset+3] = -40.0            # V ≈ -40 mV (dark)
        u0[offset+4] = 0.0              # h (I_H gate, low in dark)
        u0[offset+5] = 0.5              # Glu ≈ tonic release
    end
    # Similar for cones...
    
    # ON-Bipolar: hyperpolarized in dark (high glutamate → mGluR6 active)
    for i in 1:col.n_on
        offset = sidx.on_bc[1] + (i-1) * 4
        u0[offset]   = -60.0   # V (hyperpolarized, TRPM1 closed)
        u0[offset+1] = 0.0     # w
        u0[offset+2] = 0.8     # S_mGluR6 (high, glutamate is high)
        u0[offset+3] = 0.1     # Glu release (low, cell hyperpolarized)
    end
    
    # OFF-Bipolar: depolarized in dark (receiving glutamate)
    for i in 1:col.n_off
        offset = sidx.off_bc[1] + (i-1) * 4
        u0[offset]   = -40.0   # V (somewhat depolarized)
        u0[offset+1] = 0.2     # w
        u0[offset+2] = 0.5     # s_Glu (tracking PR glutamate)
        u0[offset+3] = 0.3     # Glu release
    end
    
    # Amacrines: near resting
    for i in 1:col.n_a2
        offset = sidx.a2[1] + (i-1) * 3
        u0[offset]   = -60.0   # V
        u0[offset+1] = 0.0     # w
        u0[offset+2] = 0.0     # Gly
    end
    
    # ... similar for other cell types
    
    # Müller: resting K+ levels
    for i in 1:col.n_muller
        offset = sidx.muller[1] + (i-1) * 4
        u0[offset]   = -80.0   # V_M (highly K+ permeable)
        u0[offset+1] = col.muller_params.K_o_rest  # K+_o endfoot
        u0[offset+2] = col.muller_params.K_o_rest  # K+_o stalk
        u0[offset+3] = 0.0     # Glu_o
    end
    
    # RPE: resting
    for i in 1:col.n_rpe
        offset = sidx.rpe[1] + (i-1) * 2
        u0[offset]   = -60.0   # V_RPE
        u0[offset+1] = col.rpe_params.K_sub_rest  # K+_sub
    end
    
    return u0
end
```

### 7.7 ERG Computation

```julia
"""
    compute_erg(sol, col, sidx)

Compute ERG field potential and per-component decomposition from solution.
"""
function compute_erg(sol, col::RetinalColumn, sidx::StateIndex)
    n_t = length(sol.t)
    erg = zeros(n_t)
    w = col.erg_weights
    
    components = Dict{Symbol, Vector{Float64}}(
        :a_wave => zeros(n_t),
        :b_wave => zeros(n_t),
        :d_wave => zeros(n_t),
        :OPs => zeros(n_t),
        :P3 => zeros(n_t),
        :c_wave => zeros(n_t),
        :ganglion => zeros(n_t)
    )
    
    for ti in 1:n_t
        u = sol.u[ti]
        
        # Photoreceptor contribution (a-wave)
        I_rod = sum_transmembrane_current(u, sidx.rod, 6, col.rod_params)
        I_cone = sum_transmembrane_current(u, sidx.cone, 6, col.cone_params)
        pr_contribution = w.rod * col.n_rod * I_rod + w.cone * col.n_cone * I_cone
        components[:a_wave][ti] = pr_contribution
        
        # ON-bipolar (b-wave)
        I_on = sum_transmembrane_current_ml(u, sidx.on_bc, 4, col.on_params)
        bc_on_contribution = w.on_bc * col.n_on * I_on
        components[:b_wave][ti] = bc_on_contribution
        
        # OFF-bipolar (d-wave)
        I_off = sum_transmembrane_current_ml(u, sidx.off_bc, 4, col.off_params)
        bc_off_contribution = w.off_bc * col.n_off * I_off
        components[:d_wave][ti] = bc_off_contribution
        
        # Amacrine OPs
        I_a2 = sum_transmembrane_current_ml(u, sidx.a2, 3, col.a2_params)
        I_gaba = sum_transmembrane_current_ml(u, sidx.gaba_ac, 3, col.gaba_params)
        op_contribution = w.a2 * col.n_a2 * I_a2 + w.gaba * col.n_gaba * I_gaba
        components[:OPs][ti] = op_contribution
        
        # Müller (P3)
        I_muller = muller_transmembrane_current(u, sidx.muller, col.muller_params)
        p3_contribution = w.muller * col.n_muller * I_muller
        components[:P3][ti] = p3_contribution
        
        # RPE (c-wave)
        I_rpe = rpe_transmembrane_current(u, sidx.rpe, col.rpe_params)
        cwave_contribution = w.rpe * col.n_rpe * I_rpe
        components[:c_wave][ti] = cwave_contribution
        
        # Ganglion
        I_gc = sum_transmembrane_current_ml(u, sidx.gc, 2, col.gc_params)
        components[:ganglion][ti] = w.gc * col.n_gc * I_gc
        
        # Total ERG
        erg[ti] = pr_contribution + bc_on_contribution + bc_off_contribution + 
                  op_contribution + p3_contribution + cwave_contribution + 
                  components[:ganglion][ti]
    end
    
    return erg, components
end
```

### 7.8 Dependencies (Project.toml)

```toml
[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7caa7"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"  # Visualization
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"       # For OP bandpass filtering
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
```

---

## 8. Implementation Roadmap

### Phase 1: Core Retinal Column (Weeks 1–3)

**Week 1: Foundation**
1. Set up Julia project structure and dependencies
2. Implement `types.jl` — all struct definitions
3. Implement `parameters.jl` — default parameter sets for rods, cones, and all ML cells
4. Implement photoreceptor phototransduction cascade (test in isolation)
5. Implement stimulus model (single flash)

**Week 2: Cell Types**
1. Implement ML dynamics for all neural cell types
2. Implement mGluR6 sign-inverting synapse for ON-bipolar
3. Implement neurotransmitter release for all types
4. Implement A2↔GABAergic reciprocal inhibition network
5. Implement Müller glia K⁺ buffering
6. Implement RPE c-wave generator

**Week 3: Integration**
1. Assemble full ODE system (`retinal_column_rhs!`)
2. Build `StateIndex` and state vector management
3. Implement `dark_adapted_state` initial conditions
4. Implement ERG field potential calculation
5. Build basic visualization (time courses, ERG decomposition)
6. Validate: single flash at 3+ intensities, verify a-wave/b-wave/OP/c-wave presence

### Phase 2: Parameter Fitting (Weeks 4–6)

1. Collect target ERG waveforms from literature (scotopic, photopic, mixed)
2. Define loss function: weighted sum of temporal feature errors
   - a-wave amplitude and implicit time
   - b-wave amplitude and implicit time
   - OP frequency and amplitude
   - c-wave amplitude and peak time
3. Implement parameter optimization using Optimization.jl or BlackBoxOptim.jl
4. Fit to Matt's own ERG data (if available)
5. Sensitivity analysis: which parameters most affect which ERG features

### Phase 3: Spatial Grid Expansion (Weeks 7–10)

1. Replicate retinal columns across 2D grid
2. Enable horizontal cell lateral connections (gap junctions)
3. Implement spatially varying stimuli
4. Add ganglion cell receptive field structure (center-surround)
5. Validate with spatially structured stimuli (gratings, spots)

---

## 9. Testing & Validation Strategy

### 9.1 Unit Tests by Cell Type

| Cell Type | Test | Expected Outcome |
|-----------|------|-----------------|
| Rod photoreceptor | Dim flash response | ~1 pA single-photon response, peak at ~200 ms |
| Rod photoreceptor | Saturating flash | Deep hyperpolarization with "nose" (I_H rebound at ~50 ms) |
| Cone photoreceptor | Bright flash | Faster kinetics, smaller amplitude per photon |
| ON-bipolar (mGluR6) | Glutamate step decrease | Depolarization (sign inversion verified) |
| ON-bipolar (mGluR6) | Dark → light transition | Cell depolarizes as PR glutamate drops |
| OFF-bipolar | Glutamate step decrease | Hyperpolarization (direct tracking) |
| A2 + GABA network | Sustained excitatory input | Oscillations at 100–160 Hz |
| Müller glia | K⁺ step increase | Depolarization, K⁺ redistribution |
| RPE | Sustained K⁺ decrease | Slow (seconds) hyperpolarization |

### 9.2 ERG Waveform Validation

**Scotopic flash (dim → saturating):**
- a-wave: negative, onset <20 ms, amplitude scales with intensity
- b-wave: positive, onset 30–60 ms, amplitude saturates (Naka-Rushton)
- OPs: visible on b-wave ascending limb at moderate intensities (bandpass 75–300 Hz)
- I_H nose: visible at saturating intensities as a transient notch at ~50 ms
- P3: slow negative component, time constant ~200–500 ms
- c-wave: slow positive, peaks at 2–5 seconds

**Photopic flash:**
- Faster a-wave (cone-driven)
- Smaller a-wave amplitude (fewer cones)
- Prominent d-wave at flash offset (OFF-pathway)
- OPs present but different frequency/amplitude profile

**Intensity-response functions:**
- a-wave amplitude vs. log intensity: sigmoidal (Naka-Rushton)
- b-wave amplitude vs. log intensity: sigmoidal, steeper than a-wave
- b/a ratio: increases at low intensities, decreases at very high intensities

### 9.3 Sensitivity Analysis

Key parameter groups to explore:

1. **Phototransduction gain** ($\gamma$, $\eta$): affects a-wave amplitude and sensitivity
2. **mGluR6 time constant** ($\tau_{mGluR6}$): affects b-wave implicit time
3. **A2/GABA coupling strengths**: controls OP frequency and amplitude
4. **NT clearance rates** ($\tau_{Gly}$, $\tau_{GABA}$): shifts OP frequency
5. **I_H parameters** ($\tau_H$, $g_H$): controls nose timing and amplitude
6. **ERG weights**: linear scaling, fit to match absolute component amplitudes
7. **RPE time constant**: controls c-wave timing

### 9.4 Performance Benchmark

Target: single flash response (5 seconds simulation, dt_save=0.1 ms) in <10 seconds.

With ~193 ODEs and `Tsit5()`, this should easily achieve sub-second solve times on a modern laptop. If stiffness is an issue (likely due to fast OP dynamics + slow RPE), switch to `TRBDF2()` or `KenCarp4()`.

---

## 10. Parameter Tables

### 10.1 Morris-Lecar Parameters by Cell Type

| Parameter | HC | ON-BC | OFF-BC | A2 | GABA-AC | DA-AC | GC | Units |
|-----------|-------|-------|--------|------|---------|-------|------|-------|
| $C_m$ | 20 | 20 | 20 | 10 | 10 | 20 | 25 | pF |
| $g_L$ | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | nS |
| $g_{Ca}$ | 4.0 | 4.0 | 4.0 | 8.0 | 8.0 | 4.0 | 5.0 | nS |
| $g_K$ | 8.0 | 8.0 | 8.0 | 12.0 | 12.0 | 8.0 | 10.0 | nS |
| $E_L$ | -60 | -60 | -50 | -60 | -60 | -60 | -65 | mV |
| $E_{Ca}$ | 120 | 120 | 120 | 120 | 120 | 120 | 120 | mV |
| $E_K$ | -84 | -84 | -84 | -84 | -84 | -84 | -84 | mV |
| $V_1$ | -1.2 | -1.2 | -1.2 | -1.2 | -1.2 | -1.2 | -1.2 | mV |
| $V_2$ | 18 | 18 | 18 | 18 | 18 | 18 | 18 | mV |
| $V_3$ | 12 | 12 | 2 | -10 | -8 | 12 | 2 | mV |
| $V_4$ | 17 | 17 | 17 | 12 | 12 | 17 | 17 | mV |
| $\phi$ | 0.067 | 0.067 | 0.067 | 0.2 | 0.15 | 0.067 | 0.04 | — |

**Notes:**
- A2 and GABA-AC have higher $g_{Ca}$, $g_K$, and $\phi$ for fast dynamics (OP generation)
- A2 has lower $C_m$ for faster membrane time constant
- $V_3$ is shifted negative for amacrines to enable oscillatory behavior near resting potential
- These are starting values; all will be tuned during Phase 2

### 10.2 Phototransduction Parameters

See §3.1.4 for rod vs. cone table.

### 10.3 Synaptic Parameters

See §4.1 connection matrix table.

---

## 11. Literature References

### Phototransduction Cascade
1. **Lamb TD, Pugh EN Jr** (1992). A quantitative account of the activation steps involved in phototransduction in amphibian photoreceptors. *J Physiol* 449:719-758.
2. **Hamer RD, Nicholas SC, Tranchina D, Lamb TD, Jarvinen JLP** (2005). Toward a unified model of vertebrate rod phototransduction. *Visual Neuroscience* 22:417-436.
3. **Dell'Orco D, Schmidt H, Bhatt DK** (2009). Phototransduction cascade model. *Biophysical Chemistry*.

### Morris-Lecar & Retinal Modeling
4. **Morris C, Lecar H** (1981). Voltage oscillations in the barnacle giant muscle fiber. *Biophys J* 35:193-213.
5. **Koz M et al.** — "Modelling cholinergic retinal waves" (*Scientific Reports*). *(Primary reference for mGluR6 implementation and ML parameter sets.)*
6. **Koz M et al.** — Prior Morris-Lecar retinal modeling work. *(Reference for cell-type-specific parameter sets.)*

### ERG & Field Potential
7. **Frishman LJ** (2006). Origins of the electroretinogram. In: Heckenlively JR, Arden GB, eds. *Principles and Practice of Clinical Electrophysiology of Vision*, 2nd ed.
8. **Robson JG, Frishman LJ** (2014). The rod-driven a-wave of the dark-adapted mammalian electroretinogram. *Prog Retin Eye Res* 39:1-22.
9. **Perlman I** (2001). The electroretinogram: ERG. *Webvision*.

### Oscillatory Potentials
10. **Wachtmeister L** (1998). Oscillatory potentials in the retina: what do they reveal? *Prog Retin Eye Res* 17:485-521.
11. **Kenyon GT, Moore B, Jeffs J, Denning KS, Stephens GJ, Travis BJ, George JS, Theiler J, Marshak DW** (2003). A model of high-frequency oscillatory potentials in retinal ganglion cells. *Visual Neuroscience* 20:465-480.

### Müller Glia
12. **Newman EA, Reichenbach A** (1996). The Müller cell: a functional element of the retina. *Trends Neurosci* 19:307-312.
13. **Kofuji P, Newman EA** (2004). Potassium buffering in the central nervous system. *Neuroscience* 129:1045-1056.

### RPE & c-wave
14. **Steinberg RH, Linsenmeier RA, Griff ER** (1983). Three light-evoked responses of the retinal pigment epithelium. *Vision Res* 23:1315-1323.
15. **Wu J, Peachey NS, Bhatt DK** — (Relevant to Dr. Koz's experimental finding that c-wave is NOT Müller-generated.)

### iH Channels in Photoreceptors
16. **Demontis GC, Longoni B, Barcaro U, Bhatt DK** (1999). Properties and functional roles of hyperpolarization-gated currents in guinea-pig retinal rods. *J Physiol* 515:813-828.
17. **Barrow AJ, Wu SM** (2009). Complementary conductance changes by IKx and Ih contribute to membrane impedance of turtle photoreceptors. *J Gen Physiol* 134:369-381.

---

## 12. Open Questions & Future Extensions

### 12.1 Müller Cell K⁺ Buffering Details

The spatial distribution of Kir channels along the Müller cell (endfoot vs. soma vs. microvilli) is critical for the P3 waveform shape. Current model uses a 2-compartment approximation. Future work:
- 3+ compartment spatial model along the Müller cell axis
- Detailed Kir channel subtypes (Kir4.1 vs. Kir2.1) with different voltage-dependence
- Aquaporin-4 co-localization effects on volume regulation

### 12.2 Horizontal Cell Connectivity

Phase 2 priority. Key unknowns:
- Gap junction conductance values for the HC network
- Whether HC feedback is primarily ephaptic, pH-mediated, or GABA-mediated (likely all three)
- Spatial extent of HC receptive fields in different species

### 12.3 RPE Mechanism Elaboration

Current model is highly simplified. Future additions:
- Cl⁻ channel dynamics on basolateral membrane
- RPE pump currents (Na⁺/K⁺-ATPase)
- Best vitelliform macular dystrophy (Best1 channel) simulation
- Light peak / dark trough (EOG components)

### 12.4 Calcium Imaging Integration

Matt has calcium imaging data. Integration pathway:
- Map simulated $[Ca^{2+}]_i$ to fluorescence signals (GCaMP kinetics)
- Forward model: $F/F_0 = f([Ca], K_d, n_{Hill}, \tau_{indicator})$
- Enable direct comparison of simulation with 2-photon calcium imaging data
- Extend to voltage imaging (ASAP family indicators) with appropriate filtering

### 12.5 Disease Modeling

The modular architecture naturally supports disease simulation:
- **Retinitis Pigmentosa**: reduce rod population, alter cone survival dynamics
- **Diabetic Retinopathy**: modify vascular parameters (connect to VesselDigitalTwin), OP changes
- **Glaucoma**: ganglion cell loss, pattern ERG changes
- **CSNB (Congenital Stationary Night Blindness)**: modify mGluR6/TRPM1 parameters (null b-wave)
- **Drug effects**: modify specific conductances (e.g., APB blocks mGluR6)

### 12.6 Connection to KozAI Platform

This retinal digital twin can serve as a template simulation on the BioSchema/KozAI marketplace:
- Cell types map to BioSchema `Cell` primitives
- Synaptic connections map to BioSchema `Connection` primitives
- Extracellular signaling maps to BioSchema `Field` primitives
- Users could swap in different photoreceptor models, add cell types, or modify connectivity
- Parameter sets become shareable/trainable assets

### 12.7 Noise and Variability

Not included in Phase 1 but important for realistic ERG:
- Photon shot noise (Poisson process on $\Phi$)
- Spontaneous isomerization in rods ("dark noise")
- Channel noise (stochastic ML, if needed)
- Population heterogeneity (parameter distributions across cells of same type)

### 12.8 Multi-Species Support

Parameter sets for different species:
- Mouse (most common ERG model)
- Human (clinical relevance)
- Macaque (primate model)
- Zebrafish (genetic models)

Different rod:cone ratios, kinetics, and ERG waveform shapes.

---

## Appendix A: Naka-Rushton Validation Targets

The intensity-response relationship for the b-wave should follow the Naka-Rushton function:

$$
V(I) = \frac{V_{max} \cdot I^n}{I^n + K^n}
$$

where:
- $V_{max}$: maximum b-wave amplitude
- $K$: semi-saturation intensity (intensity at half-max)
- $n$: Hill coefficient (slope, typically 0.7–1.2)

The a-wave follows a similar function with different $K$ and $n$.

This is a key validation target: plot simulated b-wave amplitude vs. log intensity and fit Naka-Rushton.

---

## Appendix B: OP Extraction Protocol

To extract OPs from the simulated ERG:

1. **Bandpass filter**: 75–300 Hz (4th-order Butterworth)
2. **Window**: 15–50 ms after flash onset
3. **Measure**: 
   - Number of OP wavelets (typically 4–6)
   - Individual OP amplitudes (OP1 through OP5)
   - Inter-OP interval (should be 6–10 ms, corresponding to 100–160 Hz)
   - Sum of OP amplitudes (clinical metric: ΣOP)

```julia
using DSP

function extract_ops(erg, t; fs=10000.0)
    # Bandpass 75-300 Hz
    bp = digitalfilter(Bandpass(75.0, 300.0; fs=fs), Butterworth(4))
    ops = filtfilt(bp, erg)
    
    # Window to OP region
    mask = (t .>= 15.0) .& (t .<= 50.0)
    ops_windowed = ops[mask]
    t_windowed = t[mask]
    
    return ops_windowed, t_windowed
end
```

---

## Appendix C: Complete State Variable Summary

For quick reference, the complete state vector layout:

```
Photoreceptor (rod/cone): [R*, G, Ca, V, h, Glu]          — 6 vars
Horizontal Cell:          [V, w, s_Glu]                     — 3 vars
ON-Bipolar Cell:          [V, w, S_mGluR6, Glu_release]     — 4 vars
OFF-Bipolar Cell:         [V, w, s_Glu, Glu_release]        — 4 vars
A2 Amacrine:              [V, w, Gly]                        — 3 vars
GABAergic Amacrine:       [V, w, GABA]                       — 3 vars
Dopaminergic Amacrine:    [V, w, DA]                         — 3 vars
Ganglion Cell:            [V, w]                              — 2 vars
Müller Glia:              [V_M, K_o_end, K_o_stalk, Glu_o]  — 4 vars
RPE:                      [V_RPE, K_sub]                      — 2 vars
```
