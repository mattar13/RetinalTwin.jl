# ============================================================
# off_bipolar.jl - OFF-Bipolar cell dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const OFFBC_STATE_VARS = 4
const OFFBC_V_INDEX = 1
const OFFBC_W_INDEX = 2
const OFFBC_S_GLU_INDEX = 3  # Ionotropic Glu gating
const OFFBC_GLU_INDEX = 4

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_off_bc_params()

Return default parameters for the OFF bipolar cell model as a named tuple.
Parameters are loaded from off_bipolar_params.csv.
"""
function default_off_bc_params()
    return default_off_bc_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    off_bipolar_dark_state(params)

Return dark-adapted initial conditions for an OFF bipolar cell.

# Arguments
- `params`: named tuple from `default_off_bc_params()`

# Returns
- 4-element state vector [V, w, s_Glu, Glu_release]
"""
function off_bipolar_dark_state(params)
    u0 = zeros(OFFBC_STATE_VARS)
    u0[OFFBC_V_INDEX] = -60.0      # Resting potential
    u0[OFFBC_W_INDEX] = 0.01       # Small recovery variable
    u0[OFFBC_S_GLU_INDEX] = 0.0    # Ionotropic Glu gating
    u0[OFFBC_GLU_INDEX] = 0.0      # No glutamate release at rest
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

# Use shared ML functions from horizontal.jl
# m_inf_ml(V, V1, V2), w_inf_ml(V, V3, V4), tau_w_ml(V, V3, V4)

# ── 4. Mathematical Model ───────────────────────────────────

"""
    off_bipolar_model!(du, u, p, t)

Morris-Lecar OFF bipolar cell model with ionotropic glutamate receptor.

# Arguments
- `du`: derivative vector (4 elements)
- `u`: state vector (4 elements)
- `p`: tuple `(params, glu_mean, I_inh, I_mod)` where:
  - `params`: named tuple from `default_off_bc_params()`
  - `glu_mean`: mean glutamate concentration from photoreceptors
  - `I_inh`: inhibitory synaptic current (pA)
  - `I_mod`: modulatory current (pA)
- `t`: time (ms)

# State vector
`u = [V, w, s_Glu, Glu_release]`

# Notes
Ionotropic synapse: depolarizes when glutamate is high (dark),
hyperpolarizes when Glu drops (light).
"""
function off_bipolar_model!(du, u, p, t)
    params, glu_mean, I_inh, I_mod = p

    # Decompose state vector using tuple unpacking
    V, w, s_Glu, Glu_rel = u

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
    alpha_Glu = params.alpha_Glu
    V_Glu_half = params.V_Glu_half
    V_Glu_slope = params.V_Glu_slope
    tau_Glu = params.tau_Glu

    # Ionotropic glutamate gating (fast, tau ~ 3 ms)
    tau_iGluR = 3.0
    ds_Glu = (glu_mean - s_Glu) / tau_iGluR

    # Excitatory synaptic current from photoreceptor glutamate
    g_iGluR = 4.0  # nS (from connection table)
    E_E = 0.0      # mV
    I_Glu = g_iGluR * s_Glu * (V - E_E)

    # Morris-Lecar activation functions
    m_inf = m_inf_ml(V, V1, V2)
    w_inf = w_inf_ml(V, V3, V4)
    tau_w = tau_w_ml(V, V3, V4)

    # Membrane currents
    I_L = g_L * (V - E_L)
    I_Ca = g_Ca * m_inf * (V - E_Ca)
    I_K = g_K * w * (V - E_K)

    # Derivatives
    dV = (-I_L - I_Ca - I_K + I_Glu + I_inh + I_mod) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)

    # Glutamate release (graded, voltage-dependent)
    R_glu = 1.0 / (1.0 + exp(-(V - V_Glu_half) / V_Glu_slope))
    dGlu_rel = (alpha_Glu * R_glu - Glu_rel) / tau_Glu

    # Assign derivatives
    du .= [dV, dw, ds_Glu, dGlu_rel]

    return nothing
end