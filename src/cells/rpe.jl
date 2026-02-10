# ============================================================
# rpe.jl - Retinal Pigment Epithelium
# ============================================================

# ── State indices ───────────────────────────────────────────

const RPE_STATE_VARS = 2
const RPE_V_INDEX = 1
const RPE_K_SUB_INDEX = 2

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_rpe_params()

Return default parameters for the RPE cell model as a named tuple.
Parameters are loaded from rpe_params.csv.
"""
function default_rpe_params()
    return default_rpe_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    rpe_dark_state(params)

Return dark-adapted initial conditions for an RPE cell.

# Arguments
- `params`: named tuple from `default_rpe_params()`

# Returns
- 2-element state vector [V_RPE, K_sub]
"""
function rpe_dark_state(params)
    u0 = zeros(RPE_STATE_VARS)
    u0[RPE_V_INDEX] = -70.0           # Resting potential
    u0[RPE_K_SUB_INDEX] = params.K_sub_rest  # Resting subretinal K+
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

"""
    rpe_transmembrane_current(u, params)

Compute RPE transmembrane current for ERG contribution.
"""
function rpe_transmembrane_current(u, params::NamedTuple)
    V_RPE = u[RPE_V_INDEX]
    K_sub = u[RPE_K_SUB_INDEX]

    E_K_sub = nernst_K(K_sub, params.K_i)

    return params.g_K_apical * (V_RPE - E_K_sub) +
           params.g_Cl_baso * (V_RPE - params.E_Cl) +
           params.g_L_RPE * (V_RPE - params.E_L_RPE)
end

# ── 4. Mathematical Model ───────────────────────────────────

"""
    rpe_model!(du, u, p, t)

RPE cell model for c-wave generation.

# Arguments
- `du`: derivative vector (2 elements)
- `u`: state vector (2 elements)
- `p`: tuple `(params, I_K_PR)` where:
  - `params`: named tuple from `default_rpe_params()`
  - `I_K_PR`: total K+ current from photoreceptors (pA)
- `t`: time (ms)

# State vector
`u = [V_RPE, K_sub]`

# Description
Light → reduced PR dark current → less K+ efflux → K_sub drops → RPE hyperpolarizes
This generates the slow c-wave component of the ERG.
"""
function rpe_model!(du, u, p, t)
    params, I_K_PR = p

    # Decompose state vector
    V_RPE, K_sub = u

    # Extract parameters
    tau_RPE = params.tau_RPE
    g_K_apical = params.g_K_apical
    g_Cl_baso = params.g_Cl_baso
    g_L_RPE = params.g_L_RPE
    E_Cl = params.E_Cl
    E_L_RPE = params.E_L_RPE
    K_sub_rest = params.K_sub_rest
    k_RPE = params.k_RPE
    alpha_K_RPE = params.alpha_K_RPE
    K_i = params.K_i

    # Subretinal K+ dynamics
    J_K_PR = alpha_K_RPE * I_K_PR
    dK_sub = J_K_PR - k_RPE * (K_sub - K_sub_rest)

    # RPE potential (very slow dynamics)
    E_K_sub = nernst_K(K_sub, K_i)
    dV_RPE = (-g_K_apical * (V_RPE - E_K_sub) -
              g_Cl_baso * (V_RPE - E_Cl) -
              g_L_RPE * (V_RPE - E_L_RPE)) / tau_RPE

    # Assign derivatives
    du .= [dV_RPE, dK_sub]

    return nothing
end