# Rod Photoreceptor Single-Cell Model — Explicit Parameter and Initial Condition Specification

Based on:
- Torre et al. (1990) phototransduction model
- Kamiyama et al. (1996, 2009) ionic current model
- Joslyn (2015) implementation

This document specifies a **single isolated vertebrate rod photoreceptor**.
No gap junctions. No population coupling.

---

## 1. Membrane Voltage Equation

Cm * dV/dt = - ( Iphoto + Ih + IKv + ICa + ICl(Ca) + IK(Ca) + IL + Iex + Iex2 )

### Membrane Parameters

| Parameter | Value | Units |
|----------|-------|-------|
| Cm | 0.02 | nF |

### Initial Condition

V(0) = -36.186 mV

---

## 2. Phototransduction Cascade (Outer Segment)

### State Variables

Rh, Rhi, Tr, PDE, Ca_photo, Cab_photo, cGMP

Light input:
Jhv = activated rhodopsin per second (Rh*/s)

---

### 2.1 Phototransduction ODEs

dRh/dt  = Jhv - α1*Rh + α2*Rhi  
dRhi/dt = α1*Rh - (α2 + α3)*Rhi  

dTr/dt  = ε*Rh*(Ttot - Tr) - β1*Tr + τ2*PDE - τ1*Tr*(PDEtot - PDE)  
dPDE/dt = τ1*Tr*(PDEtot - PDE) - τ2*PDE  

dCa_photo/dt  = b*J - γCa*(Ca_photo - C0)
                - k1*(eT - Cab_photo)*Ca_photo + k2*Cab_photo  

dCab_photo/dt = k1*(eT - Cab_photo)*Ca_photo - k2*Cab_photo  

dcGMP/dt = Amax / (1 + (Ca_photo / Kc)^4)
           - cGMP*(ν + σ*PDE)

---

### 2.2 Photocurrent

J = Jmax * cGMP^3 / (cGMP^3 + 10^3)

Iphoto = -J * (1 - exp((V - 8.5) / 17))

---

### 2.3 Phototransduction Parameters

| Parameter | Value | Units |
|----------|-------|-------|
| α1 | 50 | s⁻¹ |
| α2 | 0.0003 | s⁻¹ |
| α3 | 0.03 | s⁻¹ |
| ε | 0.5 | s⁻¹·µM⁻¹ |
| β1 | 2.5 | s⁻¹ |
| τ1 | 0.2 | s⁻¹·µM⁻¹ |
| τ2 | 5.0 | s⁻¹ |
| Ttot | 1000 | µM |
| PDEtot | 100 | µM |
| Jmax | 5040 | pA |
| b | 0.25 | µM·s⁻¹·pA⁻¹ |
| γCa | 50 | s⁻¹ |
| C0 | 0.1 | µM |
| k1 | 0.2 | s⁻¹·µM⁻¹ |
| k2 | 0.8 | s⁻¹ |
| eT | 500 | µM |
| Amax | 65.6 | µM·s⁻¹ |
| Kc | 0.1 | µM |
| ν | 0.4 | s⁻¹ |
| σ | 1.0 | s⁻¹·µM⁻¹ |

---

### 2.4 Phototransduction Initial Conditions (Dark State)

| Variable | Initial Value |
|---------|---------------|
| Rh | 0 |
| Rhi | 0 |
| Tr | 0 |
| PDE | 0 |
| Ca_photo | 0.3 µM |
| Cab_photo | 34.88 µM |
| cGMP | 2.0 µM |

---

## 3. Ionic Currents (Inner Segment)

---

### 3.1 Delayed Rectifier Potassium Current (IKv)

αmKv = 5*(100 - V) / (exp((100 - V)/42) - 1)  
βmKv = 9*exp(-(V - 20)/40)

αhKv = 0.15*exp(-V/22)  
βhKv = 0.4125 / (exp((40 - V)/7) + 1)

dmKv/dt = αmKv*(1 - mKv) - βmKv*mKv  
dhKv/dt = αhKv*(1 - hKv) - βhKv*hKv  

IKv = gKv * mKv^3 * hKv * (V - EK)

| Parameter | Value |
|----------|-------|
| gKv | 2.0 nS |
| EK | -74 mV |

Initial conditions:
mKv(0) = 0.430  
hKv(0) = 0.999  

---

### 3.2 Calcium Current (ICa)

αmCa = 3*(80 - V) / (exp((80 - V)/25) - 1)  
βmCa = 10 / (1 + exp((V + 38)/7))  

hCa = exp((40 - V)/18) / (1 + exp((40 - V)/18))

dmCa/dt = αmCa*(1 - mCa) - βmCa*mCa  

ICa = gCa * mCa^4 * hCa * (V - ECa)

| Parameter | Value |
|----------|-------|
| gCa | 0.7 nS |
| [Ca]o | 1600 µM |

ECa = 12.5 * log([Ca]s / [Ca]o)

Initial condition:
mCa(0) = 0.436  

---

### 3.3 Calcium-Activated Chloride Current (ICl(Ca))

mCl = 1 / (1 + exp((0.37 - [Ca]s)/0.09))

ICl(Ca) = gCl * mCl * (V - ECl)

| Parameter | Value |
|----------|-------|
| gCl | 2.0 nS |
| ECl | -20 mV |

---

### 3.4 Calcium-Activated Potassium Current (IK(Ca))

αmKCa = 15*(80 - V) / (exp((80 - V)/40) - 1)  
βmKCa = 20*exp(-V/35)

dmKCa/dt = αmKCa*(1 - mKCa) - βmKCa*mKCa  

mKCa_inf = [Ca]s / ([Ca]s + 0.3)

IK(Ca) = gKCa * mKCa^2 * mKCa_inf * (V - EK)

| Parameter | Value |
|----------|-------|
| gKCa | 5.0 nS |
| EK | -74 mV |

Initial condition:
mKCa(0) = 0.642  

---

### 3.5 Leakage Current (IL)

IL = gL * (V - EL)

| Parameter | Value |
|----------|-------|
| gL | 0.35 nS |
| EL | -77 mV |

---

## 4. Intracellular Calcium System (Inner Segment)

### Submembrane Calcium

d[Ca]s/dt = -(ICa + Iex + Iex2)/(2*F*V1)*1e-6
            - DCa*(S1/(δ*V1))*([Ca]s - [Ca]f)
            - Lb1*[Ca]s*(BL - [Cab]ls) + Lb2*[Cab]ls
            - Hb1*[Ca]s*(BH - [Cab]hs) + Hb2*[Cab]hs

Initial condition:
[Ca]s(0) = 0.0966 µM  

---

### Free Calcium

d[Ca]f/dt = DCa*(S1/(δ*V2))*([Ca]s - [Ca]f)
            - Lb1*[Ca]f*(BL - [Cab]lf) + Lb2*[Cab]lf
            - Hb1*[Ca]f*(BH - [Cab]hf) + Hb2*[Cab]hf

Initial condition:
[Ca]f(0) = 0.0966 µM  

---

### Calcium Buffers — Initial Conditions

| Variable | Initial Value (µM) |
|--------|--------------------|
| [Cab]ls | 80.929 |
| [Cab]hs | 29.068 |
| [Cab]lf | 80.929 |
| [Cab]hf | 29.068 |

---

## 5. Summary: Complete Initial State Vector

V = -36.186 mV  
mKv = 0.430  
hKv = 0.999  
mCa = 0.436  
mKCa = 0.642  
[Ca]s = 0.0966 µM  
[Ca]f = 0.0966 µM  
[Cab]ls = 80.929 µM  
[Cab]hs = 29.068 µM  
[Cab]lf = 80.929 µM  
[Cab]hf = 29.068 µM  
Rh = 0  
Rhi = 0  
Tr = 0  
PDE = 0  
Ca_photo = 0.3 µM  
Cab_photo = 34.88 µM  
cGMP = 2.0 µM  

