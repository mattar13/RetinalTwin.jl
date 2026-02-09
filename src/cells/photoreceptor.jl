# ============================================================
# photoreceptor.jl - Rod and cone photoreceptor dynamics
# ============================================================

const ROD_MKV_INDEX = 2
const ROD_HKV_INDEX = 3
const ROD_MCA_INDEX = 4
const ROD_MKCA_INDEX = 5
const ROD_CA_S_INDEX = 6
const ROD_CA_F_INDEX = 7
const ROD_CAB_LS_INDEX = 8
const ROD_CAB_HS_INDEX = 9
const ROD_CAB_LF_INDEX = 10
const ROD_CAB_HF_INDEX = 11
const ROD_RH_INDEX = 12
const ROD_RHI_INDEX = 13
const ROD_TR_INDEX = 14
const ROD_PDE_INDEX = 15
const ROD_CA_PHOTO_INDEX = 16
const ROD_CAB_PHOTO_INDEX = 17
const ROD_CGMP_INDEX = 18

const MS_PER_S = 1.0e-3

@inline function rate_fraction(scale::Real, x::Real, width::Real, limit_value::Real)
    den = exp(x / width) - 1.0
    if abs(den) < oftype(den, 1.0e-9)
        return oftype(x, limit_value)
    end
    return scale * x / den
end

@inline function release_sigmoid(V::Real, V_half::Real, V_slope::Real)
    return 1.0 / (1.0 + exp(-(V - V_half) / V_slope))
end

"""
    rod_glutamate_release(V, params::RodPhotoreceptorParams)

Steady-state glutamate release drive from rod membrane potential.
"""
function rod_glutamate_release(V::Real, params::RodPhotoreceptorParams)
    return release_sigmoid(V, params.V_Glu_half, params.V_Glu_slope)
end

"""
    rod_dark_state(params::RodPhotoreceptorParams)

Return dark-adapted initial conditions for a single rod.
"""
function rod_dark_state(params::RodPhotoreceptorParams)
    u0 = zeros(ROD_STATE_VARS)
    u0[ROD_V_INDEX] = -36.186
    u0[ROD_MKV_INDEX] = 0.430
    u0[ROD_HKV_INDEX] = 0.999
    u0[ROD_MCA_INDEX] = 0.436
    u0[ROD_MKCA_INDEX] = 0.642
    u0[ROD_CA_S_INDEX] = 0.0966
    u0[ROD_CA_F_INDEX] = 0.0966
    u0[ROD_CAB_LS_INDEX] = 80.929
    u0[ROD_CAB_HS_INDEX] = 29.068
    u0[ROD_CAB_LF_INDEX] = 80.929
    u0[ROD_CAB_HF_INDEX] = 29.068
    u0[ROD_RH_INDEX] = 0.0
    u0[ROD_RHI_INDEX] = 0.0
    u0[ROD_TR_INDEX] = 0.0
    u0[ROD_PDE_INDEX] = 0.0
    u0[ROD_CA_PHOTO_INDEX] = 0.3
    u0[ROD_CAB_PHOTO_INDEX] = 34.88
    u0[ROD_CGMP_INDEX] = 2.0
    u0[ROD_GLU_INDEX] = rod_glutamate_release(u0[ROD_V_INDEX], params)
    return u0
end

"""
    rod_rhs!(du, u, p, t)

Standalone rod RHS.
`p = (params::RodPhotoreceptorParams, stim::StimulusProtocol, I_feedback::Real)`.
"""
function rod_rhs!(du, u, p, t)
    params, stim, I_feedback = p
    Phi = compute_stimulus(stim, t)
    update_rod_photoreceptor!(du, u, params, Phi, I_feedback)
    return nothing
end

"""
    update_rod_photoreceptor!(du, u, params::RodPhotoreceptorParams, Phi, I_feedback)

Biophysical rod model:
`u = [V, mKv, hKv, mCa, mKCa, Ca_s, Ca_f, Cab_ls, Cab_hs, Cab_lf, Cab_hf, Rh, Rhi, Tr, PDE, Ca_photo, Cab_photo, cGMP, Glu]`
"""
function update_rod_photoreceptor!(du, u, params::RodPhotoreceptorParams,
                                   Phi::Real, I_feedback::Real)
    V         = u[ROD_V_INDEX]
    mKv       = u[ROD_MKV_INDEX]
    hKv       = u[ROD_HKV_INDEX]
    mCa       = u[ROD_MCA_INDEX]
    mKCa      = u[ROD_MKCA_INDEX]
    Ca_s      = u[ROD_CA_S_INDEX]
    Ca_f      = u[ROD_CA_F_INDEX]
    Cab_ls    = u[ROD_CAB_LS_INDEX]
    Cab_hs    = u[ROD_CAB_HS_INDEX]
    Cab_lf    = u[ROD_CAB_LF_INDEX]
    Cab_hf    = u[ROD_CAB_HF_INDEX]
    Rh        = u[ROD_RH_INDEX]
    Rhi       = u[ROD_RHI_INDEX]
    Tr        = u[ROD_TR_INDEX]
    PDE       = u[ROD_PDE_INDEX]
    Ca_photo  = u[ROD_CA_PHOTO_INDEX]
    Cab_photo = u[ROD_CAB_PHOTO_INDEX]
    cGMP      = u[ROD_CGMP_INDEX]
    Glu       = u[ROD_GLU_INDEX]

    Ca_s_pos = Ca_s
    Ca_f_pos = Ca_f
    Ca_photo_pos = Ca_photo
    cGMP_pos = cGMP

    # Light drive. Phi is photons/um^2/ms; convert to Rh*/s.
    Jhv = params.eta * Phi * 1000.0

    # --- Phototransduction ---
    T_free = params.T_tot - Tr
    PDE_free = params.PDE_tot - PDE
    cab_photo_free = params.eT - Cab_photo

    dRh_s = Jhv - params.alpha1 * Rh + params.alpha2 * Rhi
    dRhi_s = params.alpha1 * Rh - (params.alpha2 + params.alpha3) * Rhi
    dTr_s = params.epsilon * Rh * T_free - params.beta1 * Tr + params.tau2 * PDE - params.tau1 * Tr * PDE_free
    dPDE_s = params.tau1 * Tr * PDE_free - params.tau2 * PDE

    cGMP3 = cGMP_pos^3
    J = params.J_max * cGMP3 / (cGMP3 + 1000.0)
    Iphoto = -J * (1.0 - exp((V - 8.5) / 17.0))

    dCa_photo_s = params.b * J - params.gamma_Ca * (Ca_photo - params.C0) -
                  params.k1 * cab_photo_free * Ca_photo + params.k2 * Cab_photo
    dCab_photo_s = params.k1 * cab_photo_free * Ca_photo - params.k2 * Cab_photo
    cyclase = params.A_max / (1.0 + (Ca_photo_pos / params.K_c)^4)
    dcGMP_s = cyclase - cGMP * (params.nu + params.sigma * PDE)

    # --- Ionic currents ---
    alpha_mKv = rate_fraction(5.0, 100.0 - V, 42.0, 210.0)
    beta_mKv = 9.0 * exp(-(V - 20.0) / 40.0)
    alpha_hKv = 0.15 * exp(-V / 22.0)
    beta_hKv = 0.4125 / (exp((40.0 - V) / 7.0) + 1.0)

    alpha_mCa = rate_fraction(3.0, 80.0 - V, 25.0, 75.0)
    beta_mCa = 10.0 / (1.0 + exp((V + 38.0) / 7.0))
    hCa = 1.0 / (1.0 + exp((V - 40.0) / 18.0))

    alpha_mKCa = rate_fraction(15.0, 80.0 - V, 40.0, 600.0)
    beta_mKCa = 20.0 * exp(-V / 35.0)

    IKv = params.g_Kv * mKv^3 * hKv * (V - params.E_K)
    ECa = 12.5 * log(Ca_s_pos / params.Ca_o)
    ICa = params.g_Ca * mCa^4 * hCa * (V - ECa)
    mCl = 1.0 / (1.0 + exp((0.37 - Ca_s_pos) / 0.09))
    IClCa = params.g_Cl * mCl * (V - params.E_Cl)
    mKCa_inf = Ca_s_pos / (Ca_s_pos + 0.3)
    IKCa = params.g_KCa * mKCa^2 * mKCa_inf * (V - params.E_K)

    h_inf = 1.0 / (1.0 + exp((V - params.V_h_half) / params.k_h))
    Ih = params.g_H * h_inf * (V - params.E_H)
    IL = params.g_L * (V - params.E_L)

    Iex = params.J_ex_max * Ca_s_pos / (Ca_s_pos + params.K_ex)
    Iex2 = params.J_ex2_max * Ca_s_pos / (Ca_s_pos + params.K_ex2)

    du[ROD_V_INDEX] = (-Iphoto - Ih - IKv - ICa - IClCa - IKCa - IL - Iex - Iex2 + I_feedback) / params.C_m

    # --- Calcium compartments ---
    cab_ls_free = params.B_L - Cab_ls
    cab_hs_free = params.B_H - Cab_hs
    cab_lf_free = params.B_L - Cab_lf
    cab_hf_free = params.B_H - Cab_hf

    diffusion_s = params.D_Ca * (params.S1 / (params.delta * params.V1)) * (Ca_s - Ca_f)
    diffusion_f = params.D_Ca * (params.S1 / (params.delta * params.V2)) * (Ca_s - Ca_f)

    dCa_s_s = -(ICa + Iex + Iex2) / (2.0 * params.F * params.V1) * 1.0e-6 -
              diffusion_s -
              params.Lb1 * Ca_s_pos * cab_ls_free + params.Lb2 * Cab_ls -
              params.Hb1 * Ca_s_pos * cab_hs_free + params.Hb2 * Cab_hs

    dCa_f_s = diffusion_f -
              params.Lb1 * Ca_f_pos * cab_lf_free + params.Lb2 * Cab_lf -
              params.Hb1 * Ca_f_pos * cab_hf_free + params.Hb2 * Cab_hf

    dCab_ls_s = params.Lb1 * Ca_s_pos * cab_ls_free - params.Lb2 * Cab_ls
    dCab_hs_s = params.Hb1 * Ca_s_pos * cab_hs_free - params.Hb2 * Cab_hs
    dCab_lf_s = params.Lb1 * Ca_f_pos * cab_lf_free - params.Lb2 * Cab_lf
    dCab_hf_s = params.Hb1 * Ca_f_pos * cab_hf_free - params.Hb2 * Cab_hf

    # --- Gating and cascade derivatives (1/s -> 1/ms) ---
    du[ROD_MKV_INDEX] = (alpha_mKv * (1.0 - mKv) - beta_mKv * mKv) * MS_PER_S
    du[ROD_HKV_INDEX] = (alpha_hKv * (1.0 - hKv) - beta_hKv * hKv) * MS_PER_S
    du[ROD_MCA_INDEX] = (alpha_mCa * (1.0 - mCa) - beta_mCa * mCa) * MS_PER_S
    du[ROD_MKCA_INDEX] = (alpha_mKCa * (1.0 - mKCa) - beta_mKCa * mKCa) * MS_PER_S

    du[ROD_CA_S_INDEX] = dCa_s_s * MS_PER_S
    du[ROD_CA_F_INDEX] = dCa_f_s * MS_PER_S
    du[ROD_CAB_LS_INDEX] = dCab_ls_s * MS_PER_S
    du[ROD_CAB_HS_INDEX] = dCab_hs_s * MS_PER_S
    du[ROD_CAB_LF_INDEX] = dCab_lf_s * MS_PER_S
    du[ROD_CAB_HF_INDEX] = dCab_hf_s * MS_PER_S

    du[ROD_RH_INDEX] = dRh_s * MS_PER_S
    du[ROD_RHI_INDEX] = dRhi_s * MS_PER_S
    du[ROD_TR_INDEX] = dTr_s * MS_PER_S
    du[ROD_PDE_INDEX] = dPDE_s * MS_PER_S
    du[ROD_CA_PHOTO_INDEX] = dCa_photo_s * MS_PER_S
    du[ROD_CAB_PHOTO_INDEX] = dCab_photo_s * MS_PER_S
    du[ROD_CGMP_INDEX] = dcGMP_s * MS_PER_S

    R_glu = rod_glutamate_release(V, params)
    du[ROD_GLU_INDEX] = (params.alpha_Glu * R_glu - Glu) / params.tau_Glu

    return nothing
end

"""
    update_photoreceptor!(du, u, params::PhototransductionParams, Phi, I_feedback)

Compute derivatives for one cone photoreceptor in the reduced 6-state model.
`u = [R*, G, Ca, V, h, Glu]`.
"""
function update_photoreceptor!(du, u, params::PhototransductionParams,
                               Phi::Real, I_feedback::Real)
    R_star = u[1]
    G      = u[2]
    Ca     = u[3]
    V      = u[4]
    h      = u[5]
    Glu    = u[6]

    # Phototransduction cascade
    du[1] = params.eta * Phi - R_star / params.tau_R

    Ca_ratio = min((params.Ca_dark / Ca)^params.n_Ca, 100.0)
    du[2] = params.alpha_G * Ca_ratio - params.beta_G * (1.0 + params.gamma_PDE * R_star) * G

    G_norm = (G / params.G_dark)^params.n_G
    I_CNG = params.g_CNG_max * G_norm * (V - params.E_CNG)
    du[3] = (I_CNG * params.f_Ca - params.k_ex * Ca) / params.B_Ca

    # Membrane currents
    h_inf = 1.0 / (1.0 + exp((V - params.V_h_half) / params.k_h))
    I_H = params.g_H * h * (V - params.E_H)
    du[5] = (h_inf - h) / params.tau_H

    w_Kv = 1.0 / (1.0 + exp(-(V + 30.0) / 10.0))
    I_Kv = params.g_Kv * w_Kv * (V - params.E_K)
    du[4] = (-params.g_L * (V - params.E_L) - I_CNG - I_H - I_Kv + I_feedback) / params.C_m

    # Glutamate release
    R_glu = 1.0 / (1.0 + exp(-(V - params.V_Glu_half) / params.V_Glu_slope))
    du[6] = (params.alpha_Glu * R_glu - Glu) / params.tau_Glu

    return nothing
end

"""
    photoreceptor_K_current(u, params)

Compute K+ current for a photoreceptor for Muller/RPE K+ sensing.
"""
function photoreceptor_K_current(u, params::RodPhotoreceptorParams)
    V = u[ROD_V_INDEX]
    mKv = u[ROD_MKV_INDEX]
    hKv = u[ROD_HKV_INDEX]
    mKCa = u[ROD_MKCA_INDEX]
    Ca_s = u[ROD_CA_S_INDEX]

    IKv = params.g_Kv * mKv^3 * hKv * (V - params.E_K)
    mKCa_inf = Ca_s / (Ca_s + 0.3)
    IKCa = params.g_KCa * mKCa^2 * mKCa_inf * (V - params.E_K)
    return IKv + IKCa
end

function photoreceptor_K_current(u, params::PhototransductionParams)
    V = u[CONE_V_INDEX]
    w_Kv = 1.0 / (1.0 + exp(-(V + 30.0) / 10.0))
    return params.g_Kv * w_Kv * (V - params.E_K)
end
