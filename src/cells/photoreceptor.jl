# ============================================================
# photoreceptor.jl - Rod photoreceptor dynamics
# ============================================================

# ── 2. Initial Conditions ───────────────────────────────────

"""
    photoreceptor_state(params)

Return dark-adapted initial conditions for a rod photoreceptor.

# Arguments
- `params`: named tuple from `default_rod_params()`

# Returns
- 21-element state vector corresponding to dark-adapted equilibrium
"""
function photoreceptor_state(params)
    R0 = 0.0
    T0 = 0.0
    P0 = 0.0
    G0 = 2.0
    HC10 = 0.5
    HC20 = 0.3
    HO10 = 0.1
    HO20 = 0.05
    HO30 = 0.05
    mKv0 = 0.430
    hKv0 = 0.999
    mCa0 = 0.436
    mKCa0 = 0.642
    Ca_s0 = 0.0966
    Ca_f0 = 0.0966
    CaB_ls0 = 80.929
    CaB_hs0 = 29.068
    CaB_lf0 = 80.929
    CaB_hf0 = 29.068
    V0 = -36.186

    # Glutamate release (compute from dark voltage)
    Glu0 = R_glu_inf(V0, params)

    return [R0, T0, P0, G0, HC10, HC20, HO10, HO20, HO30, mKv0, hKv0, mCa0, mKCa0, Ca_s0, Ca_f0, CaB_ls0, CaB_hs0, CaB_lf0, CaB_hf0, V0, Glu0]
end

#The initial conditions are pretty long so here is a map of the value and it's index
const PC_IC_MAP = (
    R = 1, 
    T = 2, 
    P = 3, 
    G = 4, 
    HC1 = 5, 
    HC2 = 6, 
    HO1 = 7, 
    HO2 = 8, 
    HO3 = 9, 
    mKv = 10, 
    hKv = 11, 
    mCa = 12, 
    mKCa = 13, 
    Ca_s = 14, 
    Ca_f = 15, 
    CaB_ls = 16, 
    CaB_hs = 17, 
    CaB_lf = 18, 
    CaB_hf = 19, 
    V = 20, 
    Glu = 21
)

n_PC_STATES = length(PC_IC_MAP)
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

# Glutamate release steady-state function (voltage-dependent)
@inline R_glu_inf(V, params) = params.alpha_Glu / (1.0 + exp(-(V - params.V_Glu_half) / params.V_Glu_slope))

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
5-state Ih gating, detailed Ca dynamics, and glutamate release.

# Arguments
- `du`: derivative vector (21 elements)
- `u`: state vector (21 elements)
- `p`: tuple `(params, stim_params)` where:
  - `params`: named tuple from `default_rod_params()`
  - `stim_params`: NamedTuple with stimulus information:
    - `stim_start`: stimulus onset time (ms)
    - `stim_end`: stimulus offset time (ms)
    - `photon_flux`: photon flux (photons/µm²/ms)
    - `v_hold`: boolean, if true holds voltage at dark value
    - `I_feedback`: feedback current (pA)
- `t`: time (ms)

# State vector
`u = [R, T, P, G, HC1, HC2, HO1, HO2, HO3, mKv, hKv, mCa, mKCa,
      Ca_s, Ca_f, CaB_ls, CaB_hs, CaB_lf, CaB_hf, V, Glu]`
"""
function photoreceptor_model!(du, u, p, t)
    # Unpack parameters and stimulus info
    params, stimulus_function = p

    # Decompose state vector using tuple unpacking
    R, T, P, G, HC1, HC2, HO1, HO2, HO3, mKv, hKv, mCa, mKCa,
        Ca_s, Ca_f, CaB_ls, CaB_hs, CaB_lf, CaB_hf, V, Glu = u

    # ── Stimulus ──
    Phi = stimulus_function(t)

    # ── Reversal potentials ──
    E_LEAK = -params.ELEAK
    E_H = -params.eH
    E_K = -params.eK
    E_Cl = -params.eCl
    E_Ca = params.eCa * log(params._Ca_0 / max(Ca_s, 1e-5))

    dR = params.aC * params.lambda * Phi * (params.R_tot - R) - params.kR1 * R * (params.T_tot - T) - params.kF1 * R
    dT = params.kR1 * R * (params.T_tot - T) - params.kR2 * T * (params.P_tot - P)
    dP = params.kR2 * T * (params.P_tot - P) - params.kR3 * P
    dG = -params.kHYDRO * P * G + params.kREC * (params.G0 - G)

    # ── Currents ──
    iPHOTO = -params.iDARK * J∞(G, 10.0) * (1.0 - exp((V - 8.5) / 17.0))
    iLEAK = params.gLEAK * (V - E_LEAK)
    iH = params.gH * (HO1 + HO2 + HO3) * (V - E_H)
    iKV = params.gKV * mKv^3 * hKv * (V - E_K)
    iCa = params.gCa * mCa^4 * hCa(V) * (V - E_Ca)
    iKCa = params.gKCa * mKCa^2 * mKCas(Ca_s) * (V - E_K)
    iCl = params.gCl * mCl(Ca_s) * (V - E_Cl)
    iEX = params.J_ex * C∞(Ca_s, params.Cae, params.K_ex) * exp(-(V + 14) / 70)
    iEX2 = params.J_ex2 * C∞(Ca_s, params.Cae, params.K_ex2)

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
    Ca_flux = -(iCa + iEX + iEX2) / (2* params.F * params.V1) * 1e-6
    diffusion_s = params.DCa * (params.S1 / (params.DELTA * params.V1)) * (Ca_s - Ca_f)
    diffusion_f = params.DCa * (params.S1 / (params.DELTA * params.V2)) * (Ca_s - Ca_f)

    dCa_s = Ca_flux - diffusion_s -
            params.Lb1 * Ca_s * (params.Bl - CaB_ls) + params.Lb2 * CaB_ls -
            params.Hb1 * Ca_s * (params.Bh - CaB_hs) + params.Hb2 * CaB_hs

    dCa_f = diffusion_f -
            params.Lb1 * Ca_f * (params.Bl - CaB_lf) + params.Lb2 * CaB_lf -
            params.Hb1 * Ca_f * (params.Bh - CaB_hf) + params.Hb2 * CaB_hf

    dCaB_ls = params.Lb1 * Ca_s * (params.Bl - CaB_ls) - params.Lb2 * CaB_ls
    dCaB_hs = params.Hb1 * Ca_s * (params.Bh - CaB_hs) - params.Hb2 * CaB_hs
    dCaB_lf = params.Lb1 * Ca_f * (params.Bl - CaB_lf) - params.Lb2 * CaB_lf
    dCaB_hf = params.Hb1 * Ca_f * (params.Bh - CaB_hf) - params.Hb2 * CaB_hf

    # ── Voltage ──
    dV = (-(iPHOTO + iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2) + params.I_app) / params.C_m

    # ── Glutamate release dynamics (voltage-dependent) ──
    r_glu_inf = R_glu_inf(V, params)
    dGlu = (params.a_Glu * r_glu_inf - Glu) / params.tau_Glu

    # ── Assign all derivatives to du ──
    du .= [dR, dT, dP, dG, dHC1, dHC2, dHO1, dHO2, dHO3,
           dmKv, dhKv, dmCa, dmKCa,
           dCa_s, dCa_f, dCaB_ls, dCaB_hs, dCaB_lf, dCaB_hf,
           dV, dGlu]

    return nothing
end

"""
    photoreceptor_K_current(u, params)

Compute total K+ efflux from a rod photoreceptor.
"""
function photoreceptor_K_efflux(u, params::NamedTuple)
    V = u[PC_IC_MAP.V]
    mKv = u[PC_IC_MAP.mKv]
    hKv = u[PC_IC_MAP.hKv]
    mKCa = u[PC_IC_MAP.mKCa]
    Ca_s = u[PC_IC_MAP.Ca_s]
    E_K = -params.eK
    IKv = params.gKV * mKv^3 * hKv * (V - E_K)
    IKCa = params.gKCa * mKCa^2 * mKCas(Ca_s) * (V - E_K)

    return IKv + IKCa
end
