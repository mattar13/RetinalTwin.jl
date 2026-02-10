# ============================================================
# photoreceptor.jl - Rod photoreceptor dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const ROD_STATE_VARS = 20
const ROD_R_INDEX = 1
const ROD_T_INDEX = 2
const ROD_P_INDEX = 3
const ROD_G_INDEX = 4
const ROD_HC1_INDEX = 5
const ROD_HC2_INDEX = 6
const ROD_HO1_INDEX = 7
const ROD_HO2_INDEX = 8
const ROD_HO3_INDEX = 9
const ROD_MKV_INDEX = 10
const ROD_HKV_INDEX = 11
const ROD_MCA_INDEX = 12
const ROD_MKCA_INDEX = 13
const ROD_CA_S_INDEX = 14
const ROD_CA_F_INDEX = 15
const ROD_CAB_LS_INDEX = 16
const ROD_CAB_HS_INDEX = 17
const ROD_CAB_LF_INDEX = 18
const ROD_CAB_HF_INDEX = 19
const ROD_V_INDEX = 20

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_rod_params()

Return default parameters for the rod photoreceptor model as a named tuple.
Parameters are loaded from photoreceptor_params.csv.
"""
function default_rod_params()
    return default_rod_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    rod_dark_state(params)

Return dark-adapted initial conditions for a rod photoreceptor.

# Arguments
- `params`: named tuple from `default_rod_params()`

# Returns
- 20-element state vector corresponding to dark-adapted equilibrium
"""
function rod_dark_state(params)
    u0 = zeros(ROD_STATE_VARS)

    # Phototransduction cascade (dark = no activation)
    u0[ROD_R_INDEX] = 0.0      # R
    u0[ROD_T_INDEX] = 0.0      # T
    u0[ROD_P_INDEX] = 0.0      # P
    u0[ROD_G_INDEX] = 2.0      # G (cGMP)

    # Ih gating states (5-state model)
    u0[ROD_HC1_INDEX] = 0.5
    u0[ROD_HC2_INDEX] = 0.3
    u0[ROD_HO1_INDEX] = 0.1
    u0[ROD_HO2_INDEX] = 0.05
    u0[ROD_HO3_INDEX] = 0.05

    # Channel gating variables
    u0[ROD_MKV_INDEX] = 0.430  # mKv
    u0[ROD_HKV_INDEX] = 0.999  # hKv
    u0[ROD_MCA_INDEX] = 0.436  # mCa
    u0[ROD_MKCA_INDEX] = 0.642 # mKCa

    # Inner segment Ca dynamics
    u0[ROD_CA_S_INDEX] = 0.0966   # Ca_s
    u0[ROD_CA_F_INDEX] = 0.0966   # Ca_f
    u0[ROD_CAB_LS_INDEX] = 80.929 # CaB_ls
    u0[ROD_CAB_HS_INDEX] = 29.068 # CaB_hs
    u0[ROD_CAB_LF_INDEX] = 80.929 # CaB_lf
    u0[ROD_CAB_HF_INDEX] = 29.068 # CaB_hf

    # Membrane potential
    u0[ROD_V_INDEX] = -36.186     # V

    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

"""
    Stim(t, t_on, t_off, Phi; hold=0)

Stimulus function: returns Phi between t_on and t_off, otherwise hold value.
"""
@inline Stim(t, t_on, t_off, Phi; hold=0) = (t_on <= t <= t_off ? Phi : hold)

"""
    J∞(g, kg)

CNG channel gating function.
"""
@inline J∞(g, kg) = g^3 / (g^3 + kg^3)

"""
    C∞(C, Cae, K)

Ca exchanger saturation function (clamped at 0).
"""
@inline C∞(C, Cae, K) = C > Cae ? (C - Cae)/(C - Cae + K) : 0.0

# Voltage-gated K+ (IKv) rate functions
@inline αmKv(v) = 5*(100 - v) / (exp((100 - v)/42) - 1)
@inline βmKv(v) = 9 * exp(-(v - 20)/40)
@inline αhKv(v) = 0.15 * exp(-v/22)
@inline βhKv(v) = 0.4125 / (exp((10 - v)/7) + 1)

# L-type Ca2+ current (ICa) rate functions
@inline αmCa(v) = 3*(80 - v) / (exp((80 - v)/25) - 1)
@inline βmCa(v) = 10 / (1 + exp((v + 38)/7))
@inline hCa(v) = exp((40 - v)/18) / (1 + exp((40 - v)/18))

# Ca2+-activated K+ current (IKCa) rate functions
@inline αmKCa(v) = 15*(80 - v) / (exp((80 - v)/40) - 1)
@inline βmKCa(v) = 20 * exp(-v/35)
@inline mKCas(C) = C / (C + 0.3)

# Ca2+-activated Cl− current (ICl)
@inline mCl(C) = 1 / (1 + exp((0.37 - C)/0.09))

# Hyperpolarization-activated current (Ih) rate functions
@inline αh(v) = 8 / (exp((v + 78)/14) + 1)
@inline βh(v) = 18 / (exp(-(v + 8)/19) + 1)

"""
    hT(v)

Transition matrix for 5-state Ih gating model.
"""
function hT(v)
    α = αh(v)
    β = βh(v)
    return [
        -4α      β       0       0       0
         4α  -(3α+β)   2β       0       0
         0      3α  -(2α+2β)   3β       0
         0       0      2α  -(α+3β)   4β
         0       0       0       α    -4β
    ]
end

# ── 4. Mathematical Model ───────────────────────────────────

"""
    rod_model!(du, u, p, t)

Biophysical rod photoreceptor model with simplified phototransduction cascade,
5-state Ih gating, and detailed Ca dynamics.

# Arguments
- `du`: derivative vector (20 elements)
- `u`: state vector (20 elements)
- `p`: tuple `(params, stim_start, stim_end, photon_flux, v_hold, I_feedback)` where:
  - `params`: named tuple from `default_rod_params()`
  - `stim_start`: stimulus onset time (ms)
  - `stim_end`: stimulus offset time (ms)
  - `photon_flux`: photon flux (photons/µm²/ms)
  - `v_hold`: boolean, if true holds voltage at dark value
  - `I_feedback`: feedback current (pA)
- `t`: time (ms)

# State vector
`u = [R, T, P, G, HC1, HC2, HO1, HO2, HO3, mKv, hKv, mCa, mKCa,
      Ca_s, Ca_f, CaB_ls, CaB_hs, CaB_lf, CaB_hf, V]`
"""
function rod_model!(du, u, p, t)
    # Unpack parameters and stimulus info
    params, stim_start, stim_end, photon_flux, I_feedback = p

    # Decompose state vector using tuple unpacking
    R, T, P, G, HC1, HC2, HO1, HO2, HO3, mKv, hKv, mCa, mKCa,
        Ca_s, Ca_f, CaB_ls, CaB_hs, CaB_lf, CaB_hf, V = u

    # Extract parameters (using non-Greek names from CSV)
    aC = params.aC
    kR1 = params.kR1
    kF1 = params.kF1
    kR2 = params.kR2
    kR3 = params.kR3
    kHYDRO = params.kHYRDO
    kREC = params.kREC
    G0 = params.G0
    iDARK = params.iDARK
    kg = params.kg #TODO: Figure out why this isn't used
    C_m = params.C_m
    gLEAK = params.gLEAK
    eLEAK = params.ELEAK
    gH = params.gH
    eH = params.eH
    gKV = params.gKV
    eK = params.eK
    gCa = params.gCa
    eCa = params.eCa
    Ca_0 = params._Ca_0
    gKCa = params.gKCa
    gCl = params.gCl
    eCl = params.eCl
    F = params.F
    DCa = params.Dca
    S1 = params.S1
    DELTA = params.DELTA
    V1 = params.V1
    V2 = params.V2
    Lb1 = params.Lb1
    Bl = params.Bl
    Lb2 = params.Lb2
    Hb1 = params.Hb1
    Bh = params.Bh
    Hb2 = params.Hb2
    J_ex = params.J_ex
    Cae = params.Cae
    K_ex = params.K_ex
    J_ex2 = params.J_ex2
    K_ex2 = params.K_ex2

    # ── Stimulus ──
    Phi = Stim(t, stim_start, stim_end, photon_flux)

    # ── Reversal potentials ──
    E_LEAK = -eLEAK
    E_H = -eH
    E_K = -eK
    E_Cl = -eCl
    E_Ca = eCa * log(Ca_0 / max(Ca_s, 1e-5))

    # ── Phototransduction cascade ──
    R_tot = 3.0   # mM
    T_tot = 0.3   # mM
    P_tot = 0.021 # mM
    lambda = 0.67 # quantum efficiency

    dR = aC * lambda * Phi * (R_tot - R) - kR1 * R * (T_tot - T) - kF1 * R
    dT = kR1 * R * (T_tot - T) - kR2 * T * (P_tot - P)
    dP = kR2 * T * (P_tot - P) - kR3 * P
    dG = -kHYDRO * P * G + kREC * (G0 - G)

    # ── Currents ──
    iPHOTO = -iDARK * J∞(G, 10.0) * (1.0 - exp((V - 8.5) / 17.0))
    iLEAK = gLEAK * (V - E_LEAK)
    iH = gH * (HO1 + HO2 + HO3) * (V - E_H)
    iKV = gKV * mKv^3 * hKv * (V - E_K)
    iCa = gCa * mCa^4 * hCa(V) * (V - E_Ca)
    iKCa = gKCa * mKCa^2 * mKCas(Ca_s) * (V - E_K)
    iCl = gCl * mCl(Ca_s) * (V - E_Cl)
    iEX = J_ex * C∞(Ca_s, Cae, K_ex) * exp(-(V + 14) / 70)
    iEX2 = J_ex2 * C∞(Ca_s, Cae, K_ex2)

    # ── Hyperpolarization-activated current (Ih) gating ──
    H_vec = [HC1, HC2, HO1, HO2, HO3]
    rH = hT(V) * H_vec
    dHC1 = rH[1]
    dHC2 = rH[2]
    dHO1 = rH[3]
    dHO2 = rH[4]
    dHO3 = rH[5]

    # ── Channel gating ──
    dmKv = αmKv(V) * (1 - mKv) - βmKv(V) * mKv
    dhKv = αhKv(V) * (1 - hKv) - βhKv(V) * hKv
    dmCa = αmCa(V) * (1 - mCa) - βmCa(V) * mCa
    dmKCa = αmKCa(V) * (1 - mKCa) - βmKCa(V) * mKCa

    # ── Calcium dynamics ──
    Ca_flux = -(iCa + iEX + iEX2) / (2F * V1) * 1e-6
    diffusion_s = DCa * (S1 / (DELTA * V1)) * (Ca_s - Ca_f)
    diffusion_f = DCa * (S1 / (DELTA * V2)) * (Ca_s - Ca_f)

    dCa_s = Ca_flux - diffusion_s -
            Lb1 * Ca_s * (Bl - CaB_ls) + Lb2 * CaB_ls -
            Hb1 * Ca_s * (Bh - CaB_hs) + Hb2 * CaB_hs

    dCa_f = diffusion_f -
            Lb1 * Ca_f * (Bl - CaB_lf) + Lb2 * CaB_lf -
            Hb1 * Ca_f * (Bh - CaB_hf) + Hb2 * CaB_hf

    dCaB_ls = Lb1 * Ca_s * (Bl - CaB_ls) - Lb2 * CaB_ls
    dCaB_hs = Hb1 * Ca_s * (Bh - CaB_hs) - Hb2 * CaB_hs
    dCaB_lf = Lb1 * Ca_f * (Bl - CaB_lf) - Lb2 * CaB_lf
    dCaB_hf = Hb1 * Ca_f * (Bh - CaB_hf) - Hb2 * CaB_hf

    # ── Voltage ──
    dV = -(iPHOTO + iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2 + I_feedback) / C_m

    # ── Assign all derivatives to du ──
    du .= [dR, dT, dP, dG, dHC1, dHC2, dHO1, dHO2, dHO3,
           dmKv, dhKv, dmCa, dmKCa,
           dCa_s, dCa_f, dCaB_ls, dCaB_hs, dCaB_lf, dCaB_hf,
           dV]

    return nothing
end

"""
    photoreceptor_K_current(u, params)

Compute total K+ current from a rod photoreceptor for Müller/RPE K+ sensing.
"""
function photoreceptor_K_current(u, params::NamedTuple)
    V = u[ROD_V_INDEX]
    mKv = u[ROD_MKV_INDEX]
    hKv = u[ROD_HKV_INDEX]
    mKCa = u[ROD_MKCA_INDEX]
    Ca_s = u[ROD_CA_S_INDEX]

    eK = params.eK
    gKV = params.gKV
    gKCa = params.gKCa

    E_K = -eK
    IKv = gKV * mKv^3 * hKv * (V - E_K)
    IKCa = gKCa * mKCa^2 * mKCas(Ca_s) * (V - E_K)

    return IKv + IKCa
end
