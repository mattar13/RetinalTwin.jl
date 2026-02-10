# ============================================================
# muller.jl - Müller glial cell K+ buffering model
# ============================================================

# ── State indices ───────────────────────────────────────────

const MULLER_STATE_VARS = 4
const MULLER_V_INDEX = 1
const MULLER_K_END_INDEX = 2
const MULLER_K_STALK_INDEX = 3
const MULLER_GLU_INDEX = 4

# ── Constants ───────────────────────────────────────────────

const RT_F = 26.7  # mV at 37°C (RT/F for Nernst equation)

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_muller_params()

Return default parameters for the Müller glial cell model as a named tuple.
Parameters are loaded from muller_params.csv.
"""
function default_muller_params()
    return default_muller_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    muller_dark_state(params)

Return dark-adapted initial conditions for a Müller glial cell.

# Arguments
- `params`: named tuple from `default_muller_params()`

# Returns
- 4-element state vector [V_M, K_o_end, K_o_stalk, Glu_o]
"""
function muller_dark_state(params)
    u0 = zeros(MULLER_STATE_VARS)
    u0[MULLER_V_INDEX] = -80.0           # Hyperpolarized resting potential
    u0[MULLER_K_END_INDEX] = params.K_o_rest    # Resting K+ at endfoot
    u0[MULLER_K_STALK_INDEX] = params.K_o_rest  # Resting K+ at stalk
    u0[MULLER_GLU_INDEX] = 0.0          # No glutamate in dark
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

"""
    nernst_K(K_o, K_i)

Compute K+ Nernst potential: E_K = (RT/F) * ln(K_o / K_i).
"""
@inline nernst_K(K_o::Real, K_i::Real) = RT_F * log(max(K_o, 0.01) / K_i)

"""
    muller_transmembrane_current(u, params)

Compute Müller cell transmembrane current for ERG contribution.
"""
function muller_transmembrane_current(u, params::NamedTuple)
    V_M = u[MULLER_V_INDEX]
    K_o_end = u[MULLER_K_END_INDEX]
    K_o_stalk = u[MULLER_K_STALK_INDEX]

    E_K_end = nernst_K(K_o_end, params.K_i)
    E_K_stalk = nernst_K(K_o_stalk, params.K_i)

    return params.g_Kir_end * (V_M - E_K_end) + params.g_Kir_stalk * (V_M - E_K_stalk)
end

# ── 4. Mathematical Model ───────────────────────────────────

"""
    muller_model!(du, u, p, t)

Müller glial cell model with K+ siphoning and glutamate uptake.

# Arguments
- `du`: derivative vector (4 elements)
- `u`: state vector (4 elements)
- `p`: tuple `(params, I_K_neural, Glu_release_total)` where:
  - `params`: named tuple from `default_muller_params()`
  - `I_K_neural`: total K+ current from nearby neurons (pA)
  - `Glu_release_total`: total glutamate released by nearby neurons (µM/ms)
- `t`: time (ms)

# State vector
`u = [V_M, K_o_end, K_o_stalk, Glu_o]`
"""
function muller_model!(du, u, p, t)
    params, I_K_neural, Glu_release_total = p

    # Decompose state vector
    V_M, K_o_end, K_o_stalk, Glu_o = u

    # Extract parameters
    C_m = params.C_m
    g_Kir_end = params.g_Kir_end
    g_Kir_stalk = params.g_Kir_stalk
    K_o_rest = params.K_o_rest
    K_i = params.K_i
    tau_K_diffusion = params.tau_K_diffusion
    alpha_K = params.alpha_K
    V_max_EAAT = params.V_max_EAAT
    K_m_EAAT = params.K_m_EAAT

    # Dynamic Nernst potentials
    E_K_end = nernst_K(K_o_end, K_i)
    E_K_stalk = nernst_K(K_o_stalk, K_i)

    # Kir currents at endfoot and stalk
    I_Kir_end = g_Kir_end * (V_M - E_K_end)
    I_Kir_stalk = g_Kir_stalk * (V_M - E_K_stalk)

    # K+ uptake (with saturation)
    K_uptake_end = g_Kir_end * (K_o_end / (K_o_end + 2.0)) * (V_M - E_K_end)
    K_uptake_stalk = g_Kir_stalk * (K_o_stalk / (K_o_stalk + 2.0)) * (V_M - E_K_stalk)

    # Glutamate uptake (Michaelis-Menten)
    Glu_o_clamped = max(Glu_o, 0.0)
    J_uptake = V_max_EAAT * Glu_o_clamped / (K_m_EAAT + Glu_o_clamped)

    # Derivatives
    dV_M = (-I_Kir_end - I_Kir_stalk) / C_m

    dK_o_end = alpha_K * I_K_neural * 0.3 -  # 30% of neural K+ at endfoot
               K_uptake_end * 0.01 -
               (K_o_end - K_o_rest) / tau_K_diffusion

    dK_o_stalk = alpha_K * I_K_neural * 0.7 -  # 70% of neural K+ at stalk
                 K_uptake_stalk * 0.01 -
                 (K_o_stalk - K_o_rest) / tau_K_diffusion

    dGlu_o = Glu_release_total - J_uptake

    # Assign derivatives
    du .= [dV_M, dK_o_end, dK_o_stalk, dGlu_o]

    return nothing
end