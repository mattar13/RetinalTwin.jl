# ============================================================
# photoreceptor.jl — Phototransduction cascade + membrane dynamics
# State vector: [R*, G, Ca, V, h, Glu] (6 variables per cell)
# Spec §3.1, §7.5
# ============================================================

"""
    update_photoreceptor!(du, u, params::PhototransductionParams, Phi, I_feedback)

Compute derivatives for one photoreceptor (rod or cone).
`u = [R*, G, Ca, V, h, Glu]`, `Phi` = photon flux, `I_feedback` = HC feedback current.
"""
function update_photoreceptor!(du, u, params::PhototransductionParams,
                                Phi::Real, I_feedback::Real)
    R_star = u[1]
    G      = u[2]
    Ca     = u[3]
    V      = u[4]
    h      = u[5]
    Glu    = u[6]

    # --- Phototransduction cascade ---

    # Rhodopsin/PDE activation
    du[1] = params.eta * Phi - R_star / params.tau_R

    # cGMP dynamics with Ca feedback on guanylate cyclase
    Ca_clamped = max(Ca, 1e-6)
    Ca_ratio = (params.Ca_dark / Ca_clamped)^params.n_Ca
    G_clamped = max(G, 0.0)
    du[2] = params.alpha_G * Ca_ratio - params.beta_G * (1.0 + params.gamma_PDE * R_star) * G_clamped

    # CNG channel current (the photocurrent)
    G_norm = (G_clamped / params.G_dark)^params.n_G
    I_CNG = params.g_CNG_max * G_norm * (V - params.E_CNG)

    # Calcium dynamics
    du[3] = (I_CNG * params.f_Ca - params.k_ex * Ca_clamped) / params.B_Ca

    # --- Membrane currents ---

    # I_H: hyperpolarization-activated current (generates the "nose")
    h_inf = 1.0 / (1.0 + exp((V - params.V_h_half) / params.k_h))
    I_H = params.g_H * h * (V - params.E_H)
    du[5] = (h_inf - h) / params.tau_H

    # I_Kv: delayed rectifier (simplified steady-state)
    w_Kv = 1.0 / (1.0 + exp(-(V + 30.0) / 10.0))
    I_Kv = params.g_Kv * w_Kv * (V - params.E_K)

    # Membrane potential
    du[4] = (-params.g_L * (V - params.E_L) - I_CNG - I_H - I_Kv + I_feedback) / params.C_m

    # --- Glutamate release (graded, tonic in dark) ---
    R_glu = 1.0 / (1.0 + exp(-(V - params.V_Glu_half) / params.V_Glu_slope))
    du[6] = (params.alpha_Glu * R_glu - Glu) / params.tau_Glu

    return nothing
end

"""
    photoreceptor_K_current(u, params::PhototransductionParams)

Compute K+ current for a photoreceptor (for Müller/RPE K+ sensing).
"""
function photoreceptor_K_current(u, params::PhototransductionParams)
    V = u[4]
    w_Kv = 1.0 / (1.0 + exp(-(V + 30.0) / 10.0))
    return params.g_Kv * w_Kv * (V - params.E_K)
end
