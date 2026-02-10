# Rod Photoreceptor Population Model — Implementation Specification

Based on: Joslyn (2015), "A Mathematical Model of a Rod Photoreceptor Population", using the Kamiyama et al. (1996, 2009) single-cell ionic current model and Torre et al. (1990) phototransduction model.

---

## 1. Overview

This model describes a single vertebrate rod photoreceptor using a Hodgkin–Huxley-style parallel conductance model. The membrane potential `V` is driven by a photocurrent (from a phototransduction cascade) and several ionic currents in the inner segment. Rods can be coupled into a population via gap junctions.

The system is stiff — use an implicit/stiff ODE solver (e.g., `ode15s` in MATLAB, `scipy.integrate.solve_ivp` with method `'Radau'` or `'BDF'` in Python).

---

## 2. Membrane Voltage Equation

```
Iall = Iphoto + Ih + IKv + ICa + ICl(Ca) + IK(Ca) + IL + Iex + Iex2

Cm * dV/dt = -Iall
```

| Parameter | Value |
|-----------|-------|
| `Cm`      | 0.02 nF |

**Initial condition:** `V(0) = -36.186 mV`

---

## 3. Phototransduction Cascade (Photocurrent Model)

Source: Torre et al. (1990), as used in Kamiyama et al. (2009).

The input variable `Jhv` represents light intensity in activated rhodopsin per second (Rh\*/s).

### 3.1 ODEs

```
dRh/dt   = Jhv - α1 * Rh + α2 * Rhi
dRhi/dt  = α1 * Rh - (α2 + α3) * Rhi
dTr/dt   = ε * Rh * (Ttot - Tr) - β1 * Tr + τ2 * PDE - τ1 * Tr * (PDEtot - PDE)
dPDE/dt  = τ1 * Tr * (PDEtot - PDE) - τ2 * PDE
dCa/dt   = b * J - γCa * (Ca - C0) - k1 * (eT - Cab) * Ca + k2 * Cab
dCab/dt  = k1 * (eT - Cab) * Ca - k2 * Cab
dcGMP/dt = Amax / (1.0 + (Ca / Kc)^4) - cGMP * (V_dark + σ * PDE)
```

> **Note:** The `V` in the cGMP equation refers to a dark hydrolysis rate constant (often denoted `V_dark` or `ν` in the literature), **not** the membrane voltage. Check Kamiyama (2009) / Torre (1990) for the exact parameter name and value used.

### 3.2 Photocurrent

```
J = Jmax * cGMP^3 / (cGMP^3 + 10^3)

Iphoto = -J * (1.0 - exp((V - 8.5) / 17.0))
```

Here `V` is the membrane voltage (mV).

### 3.3 Phototransduction Parameters

These parameters come from the Torre et al. / Kamiyama et al. model. Refer to Kamiyama et al. (2009) Table/Appendix for the full set:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `α1` | Rh → Rhi forward rate | (see source) |
| `α2` | Rhi → Rh reverse rate | (see source) |
| `α3` | Rhi decay rate | (see source) |
| `ε` | Transducin activation rate | (see source) |
| `Ttot` | Total transducin | (see source) |
| `β1` | Transducin deactivation rate | (see source) |
| `τ1` | PDE activation rate | (see source) |
| `τ2` | PDE deactivation rate | (see source) |
| `PDEtot` | Total PDE | (see source) |
| `b` | Ca²⁺ influx scaling | (see source) |
| `Jmax` | Maximum CNG channel current | (see source) |
| `γCa` | Ca²⁺ extrusion rate | (see source) |
| `C0` | Resting [Ca²⁺] | (see source) |
| `k1` | Buffer binding rate | (see source) |
| `k2` | Buffer unbinding rate | (see source) |
| `eT` | Total buffer concentration | (see source) |
| `Amax` | Max guanylate cyclase rate | (see source) |
| `Kc` | Ca²⁺ half-max for cyclase | (see source) |
| `σ` | PDE catalytic rate for cGMP | (see source) |

> **Implementation note:** The phototransduction parameters above are listed in Torre et al. (1990) and Kamiyama et al. (2009). They are not fully enumerated in the Joslyn (2015) paper. You will need to pull them from Kamiyama et al. (2009), Table 1 / Appendix, or Torre et al. (1990).

---

## 4. Ionic Current Model (Inner Segment)

All currents are in pA. Voltage `V` is in mV. Conductances `g̅` are in nS. Reversal potentials `E` are in mV.

### 4.1 Delayed Rectifier Potassium Current (IKv)

```
αmKv = 5(100 - V) / (exp((100 - V) / 42) - 1)
βmKv = 9 * exp(-(V - 20) / 40)

αhKv = 0.15 * exp(-V / 22)
βhKv = 0.4125 / (exp((40 - V) / 22) + 1)

dmKv/dt = αmKv * (1 - mKv) - βmKv * mKv
dhKv/dt = αhKv * (1 - hKv) - βhKv * hKv

IKv = g̅Kv * mKv² * hKv * (V - EK)
```

| Parameter | Value |
|-----------|-------|
| `g̅Kv` | 2.0 nS |
| `EK` | -74 mV |
| `mKv(0)` | 0.430 |
| `hKv(0)` | 0.999 |

### 4.2 Calcium Current (ICa)

```
αmCa = 3(80 - V) / (exp((80 - V) / 25) - 1)
βmCa = 10 / (1 + exp((V + 38) / 7))
hCa   = exp((40 - V) / 18) / (1 + exp((40 - V) / 18))

dmCa/dt = αmCa * (1 - mCa) - βmCa * mCa

ICa = g̅Ca * mCa³ * hCa * (V - ECa)
```

| Parameter | Value |
|-----------|-------|
| `g̅Ca` | 0.7 nS |
| `ECa` | -12.5 * log([Ca]s / [Ca]o) mV (Nernst) |
| `[Ca]o` | 1600 µM |
| `mCa(0)` | 0.436 |

### 4.3 Calcium-Activated Chloride Current (ICl(Ca))

```
mCl = 1 / (1 + exp((0.37 - [Ca]s) / 0.09))

ICl(Ca) = g̅Cl * mCl * (V - ECl)
```

| Parameter | Value |
|-----------|-------|
| `g̅Cl` | 2.0 nS |
| `ECl` | -20 mV |

### 4.4 Calcium-Activated Potassium Current (IK(Ca))

```
αmKCa = 15(80 - V) / (exp((80 - V) / 40) - 1)
βmKCa = 20 * exp(-V / 35)

dmKCa/dt = αmKCa * (1 - mKCa) - βmKCa * mKCa

mKCa_inf = [Ca]s / ([Ca]s + 0.3)

IK(Ca) = g̅KCa * mKCa² * mKCa_inf * (V - EK)
```

| Parameter | Value |
|-----------|-------|
| `g̅KCa` | 5.0 nS |
| `EK` | -74 mV |
| `mKCa(0)` | 0.642 |

### 4.5 Leakage Current (IL)

```
IL = gL * (V - EL)
```

| Parameter | Value |
|-----------|-------|
| `gL` | 0.35 nS |
| `EL` | -77 mV |

### 4.6 Hyperpolarization-Activated Current (Ih)

The hyperpolarization-activated current is referenced in the paper but its full equations are in Kamiyama et al. (2009). It follows a standard HCN-channel formulation:

```
Ih = g̅h * mh * (V - Eh)
```

Refer to Kamiyama et al. (2009) for the gating variable `mh` kinetics and parameters. The sensitivity analysis identifies `Eh` as the single most influential parameter (impact = 0.1752 mV per 0.1% perturbation).

---

## 5. Intracellular Calcium System

Calcium dynamics are split between a submembrane shell compartment (`[Ca]s`) and a central (free) compartment (`[Ca]f`), each with low-affinity and high-affinity buffers.

### 5.1 Submembrane Calcium ([Ca]s)

```
d[Ca]s/dt = -(ICa + Iex + Iex2) / (2 * F * V1) * 1e-6
            - DCa * (S1 / (δ * V1)) * ([Ca]s - [Ca]f)
            - Lb1 * [Ca]s * (BL - [Cab]ls) + Lb2 * [Cab]ls
            - Hb1 * [Ca]s * (BH - [Cab]hs) + Hb2 * [Cab]hs
```

**Initial condition:** `[Ca]s(0) = 0.0966 µM`

### 5.2 Free Central Calcium ([Ca]f)

```
d[Ca]f/dt = DCa * (S1 / (δ * V2)) * ([Ca]s - [Ca]f)
            - Lb1 * [Ca]f * (BL - [Cab]lf) + Lb2 * [Cab]lf
            - Hb1 * [Ca]f * (BH - [Cab]hf) + Hb2 * [Cab]hf
```

**Initial condition:** `[Ca]f(0) = 0.0966 µM`

### 5.3 Buffer Equations

```
d[Cab]ls/dt  = Lb1 * [Ca]s * (BL - [Cab]ls)  - Lb2 * [Cab]ls
d[Cab]hs/dt  = Hb1 * [Ca]s * (BH - [Cab]hs)  - Hb2 * [Cab]hs
d[Cab]lf/dt  = Lb1 * [Ca]f * (BL - [Cab]lf)  - Lb2 * [Cab]lf
d[Cab]hf/dt  = Hb1 * [Ca]f * (BH - [Cab]hf)  - Hb2 * [Cab]hf
```

**Initial conditions:**

| Variable | Value |
|----------|-------|
| `[Cab]ls(0)` | 80.929 µM |
| `[Cab]hs(0)` | 29.068 µM |
| `[Cab]lf(0)` | 80.929 µM |
| `[Cab]hf(0)` | 29.068 µM |

### 5.4 Exchanger / Transporter Currents

```
Iex  = Jex  * exp(-(V + 14) / 70) * ([Ca]s - Cae) / ([Ca]s - Cae + Kex)
Iex2 = Jex2 * ([Ca]s - Cae) / ([Ca]s - Cae + Kex2)
```

### 5.5 Calcium System Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| `F` | 9.648 × 10⁴ | C mol⁻¹ |
| `V1` | 3.812 × 10⁻¹³ | dm³ |
| `V2` | 5.236 × 10⁻¹³ | dm³ |
| `DCa` | 6 × 10⁻⁸ | dm² s⁻¹ |
| `δ` | 3 × 10⁻⁵ | dm |
| `S1` | 3.142 × 10⁻⁸ | dm² |
| `Lb1` | 0.4 | s⁻¹ µM⁻¹ |
| `Lb2` | 0.2 | s⁻¹ |
| `Hb1` | 100 | s⁻¹ µM⁻¹ |
| `Hb2` | 90 | s⁻¹ |
| `BL` | 500 | µM |
| `BH` | 300 | µM |
| `Jex` | 20 | pA |
| `Jex2` | 20 | pA |
| `Kex` | 2.3 | µM |
| `Kex2` | 0.5 | µM |
| `Cae` | 0.01 | µM |

---

## 6. Summary of All State Variables and Initial Conditions

| # | State Variable | Description | Initial Value |
|---|---------------|-------------|---------------|
| 1 | `V` | Membrane potential (mV) | -36.186 |
| 2 | `mKv` | Kv activation gate | 0.430 |
| 3 | `hKv` | Kv inactivation gate | 0.999 |
| 4 | `mCa` | Ca activation gate | 0.436 |
| 5 | `mKCa` | K(Ca) activation gate | 0.642 |
| 6 | `[Ca]s` | Submembrane [Ca²⁺] (µM) | 0.0966 |
| 7 | `[Ca]f` | Free central [Ca²⁺] (µM) | 0.0966 |
| 8 | `[Cab]ls` | Low-affinity buffer, shell (µM) | 80.929 |
| 9 | `[Cab]hs` | High-affinity buffer, shell (µM) | 29.068 |
| 10 | `[Cab]lf` | Low-affinity buffer, central (µM) | 80.929 |
| 11 | `[Cab]hf` | High-affinity buffer, central (µM) | 29.068 |
| 12 | `Rh` | Activated rhodopsin | 0 |
| 13 | `Rhi` | Inactivated rhodopsin | 0 |
| 14 | `Tr` | Activated transducin | 0 |
| 15 | `PDE` | Activated phosphodiesterase | 0 |
| 16 | `Ca_photo` | Phototransduction Ca²⁺ | (dark steady state) |
| 17 | `Cab_photo` | Phototransduction Ca²⁺ buffer | (dark steady state) |
| 18 | `cGMP` | Cyclic GMP concentration | (dark steady state) |
| 19+ | `mh` (and possibly `hh`) | Ih gating variable(s) | (see Kamiyama 2009) |

> **Note:** The phototransduction state variables (Rh, Rhi, Tr, PDE, Ca_photo, Cab_photo, cGMP) should be initialized at their dark steady-state values (Jhv = 0). Solve the steady state algebraically or run the system in the dark until equilibrium.

---

## 7. Gap Junction Coupling (Population Model)

### 7.1 Coupling Current

For rod `i` coupled to neighboring rod `j`:

```
Igap,i = Σ_j  Ggap * (Vi - Vj)
```

This current is added to the total current for rod `i`:

```
Cm * dVi/dt = -(Iall,i + Igap,i)
```

### 7.2 Population Structures

**Cartesian grid:** Each interior cell has 4 neighbors (up, down, left, right). Boundary cells have 3 neighbors.

**Hexagonal grid (preferred):** Each interior cell has 6 neighbors. Boundary cells have 3–4 neighbors. The hexagonal layout matches the anatomical honeycomb pattern of rods in the retina and produces sharper image representations.

### 7.3 Gap Junction Conductance Matrix

Construct a sparse matrix `G` of size `N × N` (where `N` ≈ 500 cells). Entry `G[i,j] = Ggap` if rods `i` and `j` are neighbors, 0 otherwise. The matrix is symmetric.

| Parameter | Typical Values |
|-----------|---------------|
| `Ggap` | 0 (uncoupled), 1, 2, 5, 10 nS |

### 7.4 Parameter Variation Across Population

To model biological heterogeneity, each rod's parameters can be drawn from Gaussian distributions:

```
parameter_i = parameter_nominal * (1 + coefficient_of_variation * randn())
```

Typical coefficients of variation tested: 0, 1e-5, 1e-4, 1e-3, 1e-2 (i.e., 0% to 1%).

**Key finding:** Gap junction coupling reduces the spread of membrane voltage across the population when parameter noise is present. Without coupling and 10% variation, voltages range from -28 to -40 mV. With coupling, they narrow to -34 to -36 mV.

---

## 8. Sensitivity Analysis Reference

At input of 1000 Rh\*/s, steady-state voltage = -46.9305 mV. Top 10 parameters by sensitivity (0.1% perturbation):

| Rank | Parameter | ΔV (mV) |
|------|-----------|---------|
| 1 | `Eh` (Ih reversal potential) | 0.1752 |
| 2 | `EL` (leakage reversal potential) | 0.1604 |
| 3 | `EK` (K⁺ reversal potential) | 0.0694 |
| 4 | `gL` (leakage conductance) | 0.0626 |
| 5 | `Clh` | 0.0262 |
| 6 | `g̅KCa` | 0.0145 |
| 7 | `g̅Kv` | 0.0109 |
| 8 | `C0` | 0.0059 |
| 9 | `Cae` | 0.0050 |
| 10 | `ECl` | 0.0048 |

---

## 9. Simulation Protocol (Reproducing Figure 3)

To validate the implementation, reproduce the flash response figure:

1. Run the model to steady state in the dark (Jhv = 0) for sufficient time.
2. Apply 20 ms light flashes starting at t = 1.0 s.
3. Test intensities: 1, 2, 5, 10, 10, 50, 100, 200, 500, 1000 Rh\*/s.
4. Record and plot: membrane voltage (photoresponse), photocurrent, Ih, ICa, ICl(Ca), IK(Ca), d[Ca]s/dt, IL, IKv.

The voltage-intensity (V vs log Jhv) relationship should be sigmoidal, saturating around -47 mV at high intensities and resting near -36 mV in the dark.

---

## 10. Implementation Notes

- **Stiff solver required.** The calcium buffering and gating kinetics create stiffness.
- **Units consistency:** Currents in pA, voltage in mV, capacitance in nF, conductances in nS, concentrations in µM, time in seconds.
- **The Ih current** equations are not fully printed in the Joslyn paper. They must be obtained from Kamiyama et al. (1996, 2009). This is the hyperpolarization-activated cation current (HCN-type).
- **The phototransduction parameters** (α1, α2, α3, ε, Ttot, β1, τ1, τ2, PDEtot, etc.) must be obtained from Torre et al. (1990) or Kamiyama et al. (2009).
- For the **population model**, the gap junction current adds a linear coupling term to each rod's voltage equation. This increases the system size to ~19N ODEs for N rods.
