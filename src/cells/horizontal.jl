# ============================================================
# horizontal.jl - Horizontal cell dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const HC_STATE_VARS = 3
const HC_V_INDEX = 1
const HC_W_INDEX = 2
const HC_GLU_INDEX = 3

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_hc_params()

Return default parameters for the horizontal cell model as a named tuple.
Parameters are loaded from horizontal_params.csv.
"""
function default_hc_params()
    return default_hc_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    hc_dark_state(params)

Return dark-adapted initial conditions for a horizontal cell.

# Arguments
- `params`: named tuple from `default_hc_params()`

# Returns
- 3-element state vector [V, w, s_Glu]
"""
function hc_dark_state(params)
    u0 = zeros(HC_STATE_VARS)
    u0[HC_V_INDEX] = -60.0      # Resting potential
    u0[HC_W_INDEX] = 0.01       # Small recovery variable
    u0[HC_GLU_INDEX] = 0.0      # No glutamate release at rest
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

"""
    m_inf_ml(V, V1, V2)

Morris-Lecar Ca activation steady state.
"""
@inline m_inf_ml(V, V1, V2) = 0.5 * (1.0 + tanh((V - V1) / V2))

"""
    w_inf_ml(V, V3, V4)

Morris-Lecar K activation steady state.
"""
@inline w_inf_ml(V, V3, V4) = 0.5 * (1.0 + tanh((V - V3) / V4))

"""
    tau_w_ml(V, V3, V4)

Morris-Lecar K activation time constant.
"""
@inline tau_w_ml(V, V3, V4) = 1.0 / cosh((V - V3) / (2.0 * V4))

"""
    hc_feedback(V_hc; g_FB=1.0, V_FB_half=-50.0, V_FB_slope=5.0)

Compute HC feedback signal to photoreceptors.
Returns a feedback current magnitude.
"""
function hc_feedback(V_hc::Real; g_FB::Real=1.0,
                     V_FB_half::Real=-50.0, V_FB_slope::Real=5.0)
    return g_FB / (1.0 + exp(-(V_hc - V_FB_half) / V_FB_slope))
end

# ── 4. Mathematical Model ───────────────────────────────────

"""
    horizontal_model!(du, u, p, t)

Morris-Lecar horizontal cell model.

# Arguments
- `du`: derivative vector (3 elements)
- `u`: state vector (3 elements)
- `p`: tuple `(params, I_exc, I_gap, glu_mean)` where:
  - `params`: named tuple from `default_hc_params()`
  - `I_exc`: excitatory synaptic current from photoreceptors (pA)
  - `I_gap`: gap junction current from neighboring HCs (pA)
  - `glu_mean`: mean glutamate concentration (for tracking)
- `t`: time (ms)

# State vector
`u = [V, w, s_Glu]`
"""
function horizontal_model!(du, u, p, t)
    params, I_exc, I_gap, glu_mean = p

    # Decompose state vector
    V, w, s_Glu = u

    # Extract parameters
    C_m = params.C_m
    g_L = params.g_L
    g_Ca = params.g_Ca
    g_K = params.g_K
    E_L = params.E_L
    E_Ca = params.E_Ca
    E_K = params.E_K
    V1 = params.V1
    V2 = params.V2
    V3 = params.V3
    V4 = params.V4
    phi = params.phi

    # Morris-Lecar activation functions
    m_inf = m_inf_ml(V, V1, V2)
    w_inf = w_inf_ml(V, V3, V4)
    tau_w = tau_w_ml(V, V3, V4)

    # Membrane currents
    I_L = g_L * (V - E_L)
    I_Ca = g_Ca * m_inf * (V - E_Ca)
    I_K = g_K * w * (V - E_K)

    # Derivatives
    dV = (-I_L - I_Ca - I_K + I_exc + I_gap + params.I_app) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)
    ds_Glu = (glu_mean - s_Glu) / params.tau_Glu

    # Assign derivatives
    du .= [dV, dw, ds_Glu]

    return nothing
end
