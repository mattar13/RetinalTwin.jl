# ============================================================
# a2_amacrine.jl - A2 (AII) Amacrine cell dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const A2_STATE_VARS = 3
const A2_V_INDEX = 1
const A2_W_INDEX = 2
const A2_GLY_INDEX = 3

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_a2_params()

Return default parameters for the A2 amacrine cell model as a named tuple.
Parameters are loaded from a2_amacrine_params.csv.
"""
function default_a2_params()
    return default_a2_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    a2_dark_state(params)

Return dark-adapted initial conditions for an A2 amacrine cell.

# Arguments
- `params`: named tuple from `default_a2_params()`

# Returns
- 3-element state vector [V, w, Gly]
"""
function a2_dark_state(params)
    u0 = zeros(A2_STATE_VARS)
    u0[A2_V_INDEX] = -60.0      # Resting potential
    u0[A2_W_INDEX] = 0.01       # Small recovery variable
    u0[A2_GLY_INDEX] = 0.0      # No glycine release at rest
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

# Use shared ML functions from horizontal.jl
# m_inf_ml(V, V1, V2), w_inf_ml(V, V3, V4), tau_w_ml(V, V3, V4)

# ── 4. Mathematical Model ───────────────────────────────────

"""
    a2_model!(du, u, p, t)

Morris-Lecar A2 (AII) amacrine cell model.

# Arguments
- `du`: derivative vector (3 elements)
- `u`: state vector (3 elements)
- `p`: tuple `(params, I_exc, I_inh, I_mod)` where:
  - `params`: named tuple from `default_a2_params()`
  - `I_exc`: excitatory synaptic current from bipolars (pA)
  - `I_inh`: inhibitory synaptic current (pA)
  - `I_mod`: modulatory current (pA)
- `t`: time (ms)

# State vector
`u = [V, w, Gly]`

# Notes
Critical for oscillatory potential generation. Fast ML dynamics
(low C_m, high phi) enable high-frequency oscillations.
"""
function a2_model!(du, u, p, t)
    params, I_exc, I_inh, I_mod = p

    # Decompose state vector using tuple unpacking
    V, w, Gly = u

    # Extract Morris-Lecar parameters
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

    # Extract glycine release parameters
    alpha_Gly = params.alpha_Gly
    V_Gly_half = params.V_Gly_half
    V_Gly_slope = params.V_Gly_slope
    tau_Gly = params.tau_Gly

    # Morris-Lecar activation functions
    m_inf = m_inf_ml(V, V1, V2)
    w_inf = w_inf_ml(V, V3, V4)
    tau_w = tau_w_ml(V, V3, V4)

    # Membrane currents
    I_L = g_L * (V - E_L)
    I_Ca = g_Ca * m_inf * (V - E_Ca)
    I_K = g_K * w * (V - E_K)

    # Derivatives
    dV = (-I_L - I_Ca - I_K + I_exc + I_inh + I_mod) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)

    # Glycine release (fast, critical for OP frequency)
    T_inf = 1.0 / (1.0 + exp(-(V - V_Gly_half) / V_Gly_slope))
    dGly = (alpha_Gly * T_inf - Gly) / tau_Gly

    # Assign derivatives
    du .= [dV, dw, dGly]

    return nothing
end