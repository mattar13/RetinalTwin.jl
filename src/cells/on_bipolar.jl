# ============================================================
# on_bipolar.jl - ON-Bipolar cell dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const ONBC_STATE_VARS = 4
const ONBC_V_INDEX = 1
const ONBC_W_INDEX = 2
const ONBC_S_INDEX = 3  # mGluR6 state
const ONBC_GLU_INDEX = 4

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_on_bc_params()

Return default parameters for the ON bipolar cell model as a named tuple.
Parameters are loaded from on_bipolar_params.csv.
"""
function default_on_bc_params()
    return default_on_bc_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    on_bipolar_dark_state(params)

Return dark-adapted initial conditions for an ON bipolar cell.

# Arguments
- `params`: named tuple from `default_on_bc_params()`

# Returns
- 4-element state vector [V, w, S_mGluR6, Glu_release]
"""
function on_bipolar_dark_state(params)
    u0 = zeros(ONBC_STATE_VARS)
    u0[ONBC_V_INDEX] = -60.0      # Resting potential
    u0[ONBC_W_INDEX] = 0.01       # Small recovery variable
    u0[ONBC_S_INDEX] = 0.0        # mGluR6 state
    u0[ONBC_GLU_INDEX] = 0.0      # No glutamate release at rest
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

# Use shared ML functions from horizontal.jl
# m_inf_ml(V, V1, V2), w_inf_ml(V, V3, V4), tau_w_ml(V, V3, V4)

# ── 4. Mathematical Model ───────────────────────────────────

"""
    on_bipolar_model!(du, u, p, t)

Morris-Lecar ON bipolar cell model with mGluR6 sign inversion.

# Arguments
- `du`: derivative vector (4 elements)
- `u`: state vector (4 elements)
- `p`: tuple `(params, mg, glu_mean, I_inh, I_mod)` where:
  - `params`: named tuple from `default_on_bc_params()`
  - `mg`: mGluR6Params struct with mGluR6-specific parameters
  - `glu_mean`: mean glutamate concentration from photoreceptors
  - `I_inh`: inhibitory synaptic current (pA)
  - `I_mod`: modulatory current (pA)
- `t`: time (ms)

# State vector
`u = [V, w, S_mGluR6, Glu_release]`

# Notes
The mGluR6 synapse inverts the glutamate signal: high Glu → cell hyperpolarized.
TRPM1 conductance is maximal when S is low (light condition).
"""
function on_bipolar_model!(du, u, p, t)
    params, glu_received = p

    # Decompose state vector using tuple unpacking
    V, w, S, Glu_rel = u

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

    # Extract glutamate release parameters
    alpha_mGluR6 = params.alpha_mGluR6
    tau_mGluR6 = params.tau_mGluR6
    g_TRPM1 = params.g_TRPM1
    E_TRPM1 = params.E_TRPM1
    alpha_Glu = params.alpha_Glu
    V_Glu_half = params.V_Glu_half
    V_Glu_slope = params.V_Glu_slope
    tau_Glu = params.tau_Glu

    # mGluR6 cascade: tracks glutamate with metabotropic kinetics
    dS = (alpha_mGluR6 * glu_received - S) / tau_mGluR6

    # TRPM1 conductance (sign-inverted: low S → high conductance → depolarization)
    I_TRPM1 = g_TRPM1 * (1.0 - S) * (V - E_TRPM1)

    # Morris-Lecar activation functions
    m_inf = m_inf_ml(V, V1, V2)
    w_inf = w_inf_ml(V, V3, V4)
    tau_w = tau_w_ml(V, V3, V4)

    # Membrane currents
    I_L = g_L * (V - E_L)
    I_Ca = g_Ca * m_inf * (V - E_Ca)
    I_K = g_K * w * (V - E_K)

    # Derivatives
    dV = (-I_L - I_Ca - I_K - I_TRPM1 + I_inh + I_mod) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)

    # Glutamate release (graded, voltage-dependent)
    R_glu = 1.0 / (1.0 + exp(-(V - V_Glu_half) / V_Glu_slope))
    dGlu_rel = (alpha_Glu * R_glu - Glu_rel) / tau_Glu

    # Assign derivatives
    du .= [dV, dw, dS, dGlu_rel]

    return nothing
end