# ============================================================
# rpe.jl — Retinal Pigment Epithelium
# State vector: [V_RPE, K_sub] (2 variables per cell)
# Spec §3.10 — Generates the c-wave (slow positive component)
# ============================================================

"""
    update_rpe!(du, u, params::RPEParams, I_K_PR)

Compute derivatives for one RPE cell.
`u = [V_RPE, K_sub]`.
`I_K_PR` = total K+ current from photoreceptors (drives subretinal K+ changes).
"""
function update_rpe!(du, u, params::RPEParams, I_K_PR::Real)
    V_RPE = u[1]
    K_sub = u[2]

    # Subretinal K+ dynamics
    # Light → reduced PR dark current → less K+ efflux → K_sub drops → RPE hyperpolarizes
    J_K_PR = params.alpha_K_RPE * I_K_PR
    du[2] = J_K_PR - params.k_RPE * (K_sub - params.K_sub_rest)

    # RPE potential (very slow dynamics)
    E_K_sub = nernst_K(K_sub, params.K_i)
    du[1] = (-params.g_K_apical * (V_RPE - E_K_sub) -
              params.g_Cl_baso * (V_RPE - params.E_Cl) -
              params.g_L_RPE * (V_RPE - params.E_L_RPE)) / params.tau_RPE

    return nothing
end

"""
    rpe_transmembrane_current(u, params::RPEParams)

Compute RPE transmembrane current for ERG contribution.
"""
function rpe_transmembrane_current(u, params::RPEParams)
    V_RPE = u[1]
    K_sub = u[2]
    E_K_sub = nernst_K(K_sub, params.K_i)
    return params.g_K_apical * (V_RPE - E_K_sub) +
           params.g_Cl_baso * (V_RPE - params.E_Cl) +
           params.g_L_RPE * (V_RPE - params.E_L_RPE)
end
