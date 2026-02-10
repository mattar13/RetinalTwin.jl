# ============================================================
# gaba_amacrine.jl - GABAergic Amacrine cell dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const GABA_STATE_VARS = 3
const GABA_V_INDEX = 1
const GABA_W_INDEX = 2
const GABA_GABA_INDEX = 3

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_gaba_params()

Return default parameters for the GABAergic amacrine cell model as a named tuple.
Parameters are loaded from gaba_amacrine_params.csv.
"""
function default_gaba_params()
    return default_gaba_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    gaba_dark_state(params)

Return dark-adapted initial conditions for a GABAergic amacrine cell.

# Arguments
- `params`: named tuple from `default_gaba_params()`

# Returns
- 3-element state vector [V, w, GABA]
"""
function gaba_dark_state(params)
    u0 = zeros(GABA_STATE_VARS)
    u0[GABA_V_INDEX] = -60.0      # Resting potential
    u0[GABA_W_INDEX] = 0.01       # Small recovery variable
    u0[GABA_GABA_INDEX] = 0.0     # No GABA release at rest
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

# Use shared ML functions from horizontal.jl
# m_inf_ml(V, V1, V2), w_inf_ml(V, V3, V4), tau_w_ml(V, V3, V4)

# ── 4. Mathematical Model ───────────────────────────────────

"""
    gaba_model!(du, u, p, t)

Morris-Lecar GABAergic amacrine cell model.

# Arguments
- `du`: derivative vector (3 elements)
- `u`: state vector (3 elements)
- `p`: tuple `(params, I_exc, I_inh, I_mod)` where:
  - `params`: named tuple from `default_gaba_params()`
  - `I_exc`: excitatory synaptic current from bipolars (pA)
  - `I_inh`: inhibitory synaptic current (pA)
  - `I_mod`: modulatory current (pA)
- `t`: time (ms)

# State vector
`u = [V, w, GABA]`

# Notes
Forms reciprocal inhibitory network with A2 amacrines for oscillatory
potential generation.
"""
function gaba_model!(du, u, p, t)
    params, I_exc, I_inh, I_mod = p

    # Decompose state vector using tuple unpacking
    V, w, GABA = u

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

    # Extract GABA release parameters
    alpha_GABA = params.alpha_GABA
    V_GABA_half = params.V_GABA_half
    V_GABA_slope = params.V_GABA_slope
    tau_GABA = params.tau_GABA

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

    # GABA release
    T_inf = 1.0 / (1.0 + exp(-(V - V_GABA_half) / V_GABA_slope))
    dGABA = (alpha_GABA * T_inf - GABA) / tau_GABA

    # Assign derivatives
    du .= [dV, dw, dGABA]

    return nothing
end