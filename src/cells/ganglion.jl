# ============================================================
# ganglion.jl - Ganglion cell dynamics
# ============================================================

# ── State indices ───────────────────────────────────────────

const GC_STATE_VARS = 2
const GC_V_INDEX = 1
const GC_W_INDEX = 2

# ── 1. Default Parameters ───────────────────────────────────

"""
    default_gc_params()

Return default parameters for the ganglion cell model as a named tuple.
Parameters are loaded from ganglion_params.csv.
"""
function default_gc_params()
    return default_gc_params_csv()
end

# ── 2. Initial Conditions ───────────────────────────────────

"""
    ganglion_dark_state(params)

Return dark-adapted initial conditions for a ganglion cell.

# Arguments
- `params`: named tuple from `default_gc_params()`

# Returns
- 2-element state vector [V, w]
"""
function ganglion_dark_state(params)
    u0 = zeros(GC_STATE_VARS)
    u0[GC_V_INDEX] = -65.0      # Resting potential
    u0[GC_W_INDEX] = 0.01       # Small recovery variable
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

# Use shared ML functions from horizontal.jl
# m_inf_ml(V, V1, V2), w_inf_ml(V, V3, V4), tau_w_ml(V, V3, V4)

# ── 4. Mathematical Model ───────────────────────────────────

"""
    ganglion_model!(du, u, p, t)

Morris-Lecar ganglion cell model.

# Arguments
- `du`: derivative vector (2 elements)
- `u`: state vector (2 elements)
- `p`: tuple `(params, I_exc, I_inh)` where:
  - `params`: named tuple from `default_gc_params()`
  - `I_exc`: excitatory synaptic current from bipolars (pA)
  - `I_inh`: inhibitory synaptic current from amacrines (pA)
- `t`: time (ms)

# State vector
`u = [V, w]`

# Notes
Output neuron for action potential generation. Receives excitatory
input from ON/OFF bipolars and inhibitory input from amacrines.
"""
function ganglion_model!(du, u, p, t)
    params, I_exc, I_inh = p

    # Decompose state vector using tuple unpacking
    V, w = u

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

    # Morris-Lecar activation functions
    m_inf = m_inf_ml(V, V1, V2)
    w_inf = w_inf_ml(V, V3, V4)
    tau_w = tau_w_ml(V, V3, V4)

    # Membrane currents
    I_L = g_L * (V - E_L)
    I_Ca = g_Ca * m_inf * (V - E_Ca)
    I_K = g_K * w * (V - E_K)

    # Derivatives
    dV = (-I_L - I_Ca - I_K + I_exc + I_inh) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)

    # Assign derivatives
    du .= [dV, dw]

    return nothing
end