# ============================================================
# photoreceptor.jl - Rod photoreceptor dynamics
# ============================================================

"""
    photoreceptor_state(params)

Return dark-adapted initial conditions for a rod photoreceptor.
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

    Glu0 = R_glu_inf(V0, params)

    return [R0, T0, P0, G0, HC10, HC20, HO10, HO20, HO30, mKv0, hKv0, mCa0, mKCa0, Ca_s0, Ca_f0, CaB_ls0, CaB_hs0, CaB_lf0, CaB_hf0, V0, Glu0]
end

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

"""
    I_photoreceptor(V, params)

Approximate total photoreceptor transmembrane current at voltage `V` using
steady-state channel activation assumptions.
"""
function I_photoreceptor(V, params)
    Ca_ref = 0.1
    G_ref = 2.0
    mKv_inf = gate_inf(V, -35.0, 8.0)
    hKv_inf = gate_inf(V, -58.0, -8.0)
    mCa_inf = gate_inf(V, -28.0, 6.0)
    mKCa_inf = gate_inf(V, -32.0, 10.0)
    h_open = gate_inf(V, -72.0, -8.0)

    i_photo = photoreceptor_I_photo(V, G_ref, params)
    i_leak = photoreceptor_I_leak(V, params)
    i_h = photoreceptor_I_h(V, h_open, params)
    i_kv = photoreceptor_I_kv(V, mKv_inf, hKv_inf, params)
    i_ca = photoreceptor_I_ca(V, mCa_inf, Ca_ref, params)
    i_kca = photoreceptor_I_kca(V, mKCa_inf, Ca_ref, params)
    i_cl = photoreceptor_I_cl(V, Ca_ref, params)

    return i_photo + i_leak + i_h + i_kv + i_ca + i_kca + i_cl
end

"""
    photoreceptor_model!(du, u, p, t)

Biophysical rod photoreceptor model.
"""
function photoreceptor_model!(du, u, p, t)
    params, stimulus_function = p

    R, T, P, G, HC1, HC2, HO1, HO2, HO3, mKv, hKv, mCa, mKCa,
        Ca_s, Ca_f, CaB_ls, CaB_hs, CaB_lf, CaB_hf, V, Glu = u

    Phi = stimulus_function(t)

    #Model constants
    E_LEAK = -params.ELEAK
    E_H = -params.eH
    E_K = -params.eK
    E_Cl = -params.eCl
    E_Ca = params.eCa * log(params._Ca_0 / max(Ca_s, 1e-5))

    dR = params.aC * params.lambda * Phi * (params.R_tot - R) - params.kR1 * R * (params.T_tot - T) - params.kF1 * R
    dT = params.kR1 * R * (params.T_tot - T) - params.kR2 * T * (params.P_tot - P)
    dP = params.kR2 * T * (params.P_tot - P) - params.kR3 * P
    dG = -params.kHYDRO * P * G + params.kREC * (params.G0 - G)

    iPHOTO = -params.iDARK * J∞(G, 10.0) * (1.0 - exp((V - 8.5) / 17.0))
    iLEAK = params.gLEAK * (V - E_LEAK)
    iH = params.gH * (HO1 + HO2 + HO3) * (V - E_H)
    iKV = params.gKV * mKv^3 * hKv * (V - E_K)
    iCa = params.gCa * mCa^4 * hCa(V) * (V - E_Ca)
    iKCa = params.gKCa * mKCa^2 * mKCas(Ca_s) * (V - E_K)
    iCl = params.gCl * mCl(Ca_s) * (V - E_Cl)
    iEX = params.J_ex * C∞(Ca_s, params.Cae, params.K_ex) * exp(-(V + 14) / 70)
    iEX2 = params.J_ex2 * C∞(Ca_s, params.Cae, params.K_ex2)

    H_vec = [HC1, HC2, HO1, HO2, HO3]
    rH = hT(V) * H_vec
    dHC1 = rH[1]
    dHC2 = rH[2]
    dHO1 = rH[3]
    dHO2 = rH[4]
    dHO3 = rH[5]

    dmKv = αmKv(V) * (1 - mKv) - βmKv(V) * mKv
    dhKv = αhKv(V) * (1 - hKv) - βhKv(V) * hKv
    dmCa = αmCa(V) * (1 - mCa) - βmCa(V) * mCa
    dmKCa = αmKCa(V) * (1 - mKCa) - βmKCa(V) * mKCa

    F = 96485.332#Faraday constant (C/mol)
    Ca_flux = -(iCa + iEX + iEX2) / (2 * F * params.V1) * 1e-6
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

    dV = (-(iPHOTO + iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2) + params.I_app) / params.C_m

    r_glu_inf = R_glu_inf(V, params)
    dGlu = (params.a_Glu * r_glu_inf - Glu) / params.tau_Glu

    du .= [dR, dT, dP, dG, dHC1, dHC2, dHO1, dHO2, dHO3,
           dmKv, dhKv, dmCa, dmKCa,
           dCa_s, dCa_f, dCaB_ls, dCaB_hs, dCaB_lf, dCaB_hf,
           dV, dGlu]

    return nothing
end

"""
    photoreceptor_K_efflux(u, params)

Compute total K+ efflux from a rod photoreceptor.
"""
function photoreceptor_K_efflux(u, params)
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
