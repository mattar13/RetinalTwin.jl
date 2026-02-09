# ============================================================
# muller.jl — Müller glial cell K+ buffering model
# State vector: [V_M, K_o_end, K_o_stalk, Glu_o] (4 variables per cell)
# Spec §3.9 — Generates slow P3 component via K+ siphoning
# ============================================================

const RT_F = 26.7  # mV at 37°C (RT/F for Nernst equation)

"""
    nernst_K(K_o, K_i)

Compute K+ Nernst potential: E_K = (RT/F) * ln(K_o / K_i).
"""
function nernst_K(K_o::Real, K_i::Real)
    return RT_F * log(max(K_o, 0.01) / K_i)
end

"""
    update_muller!(du, u, params::MullerParams, I_K_neural, Glu_release_total)

Compute derivatives for one Müller glial cell.
`u = [V_M, K_o_end, K_o_stalk, Glu_o]`.
`I_K_neural` = total K+ current from nearby neurons (PRs + bipolars).
`Glu_release_total` = total glutamate released by nearby neurons.
"""
function update_muller!(du, u, params::MullerParams,
                        I_K_neural::Real, Glu_release_total::Real)
    V_M       = u[1]
    K_o_end   = u[2]
    K_o_stalk = u[3]
    Glu_o     = u[4]

    # Dynamic Nernst potentials
    E_K_end   = nernst_K(K_o_end, params.K_i)
    E_K_stalk = nernst_K(K_o_stalk, params.K_i)

    # Kir currents at endfoot and stalk
    I_Kir_end   = params.g_Kir_end * (V_M - E_K_end)
    I_Kir_stalk = params.g_Kir_stalk * (V_M - E_K_stalk)

    # Membrane potential (primarily K+ permeable)
    du[1] = (-I_Kir_end - I_Kir_stalk) / params.C_m

    # Extracellular K+ at endfoot (inner retina, near ganglion cells)
    # Neural K+ release accumulates here; Kir uptake removes it
    K_uptake_end = params.g_Kir_end * (K_o_end / (K_o_end + 2.0)) * (V_M - E_K_end)
    du[2] = params.alpha_K * I_K_neural * 0.3 -  # 30% of neural K+ at endfoot
            K_uptake_end * 0.01 -
            (K_o_end - params.K_o_rest) / params.tau_K_diffusion

    # Extracellular K+ at stalk (outer retina, near photoreceptors)
    K_uptake_stalk = params.g_Kir_stalk * (K_o_stalk / (K_o_stalk + 2.0)) * (V_M - E_K_stalk)
    du[3] = params.alpha_K * I_K_neural * 0.7 -  # 70% of neural K+ at stalk
            K_uptake_stalk * 0.01 -
            (K_o_stalk - params.K_o_rest) / params.tau_K_diffusion

    # Extracellular glutamate (EAAT uptake, Michaelis-Menten)
    Glu_o_clamped = max(Glu_o, 0.0)
    J_uptake = params.V_max_EAAT * Glu_o_clamped / (params.K_m_EAAT + Glu_o_clamped)
    du[4] = Glu_release_total - J_uptake

    return nothing
end

"""
    muller_transmembrane_current(u, params::MullerParams)

Compute Müller cell transmembrane current for ERG contribution.
"""
function muller_transmembrane_current(u, params::MullerParams)
    V_M       = u[1]
    K_o_end   = u[2]
    K_o_stalk = u[3]
    E_K_end   = nernst_K(K_o_end, params.K_i)
    E_K_stalk = nernst_K(K_o_stalk, params.K_i)
    return params.g_Kir_end * (V_M - E_K_end) + params.g_Kir_stalk * (V_M - E_K_stalk)
end
