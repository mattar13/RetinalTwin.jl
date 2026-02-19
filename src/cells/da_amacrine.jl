# ============================================================
# da_amacrine.jl - Dopaminergic Amacrine cell dynamics
# ============================================================

# ── 2. Initial Conditions ───────────────────────────────────

"""
    da_dark_state(params)

Return dark-adapted initial conditions for a dopaminergic amacrine cell.

# Arguments
- `params`: named tuple from `default_da_params()`

# Returns
- 3-element state vector [V, w, DA]
"""
function da_dark_state(params)
    u0 = zeros(DA_STATE_VARS)
    u0[DA_V_INDEX] = -60.0      # Resting potential
    u0[DA_W_INDEX] = 0.01       # Small recovery variable
    u0[DA_DA_INDEX] = 0.0       # No dopamine release at rest
    return u0
end

# ── 3. Auxiliary Functions ──────────────────────────────────

# Use shared ML functions from horizontal.jl
# m_inf_ml(V, V1, V2), w_inf_ml(V, V3, V4), tau_w_ml(V, V3, V4)

# ── 4. Mathematical Model ───────────────────────────────────

"""
    da_model!(du, u, p, t)

Morris-Lecar dopaminergic amacrine cell model.

# Arguments
- `du`: derivative vector (3 elements)
- `u`: state vector (3 elements)
- `p`: tuple `(params, I_exc)` where:
  - `params`: named tuple from `default_da_params()`
  - `I_exc`: excitatory synaptic current from ON-bipolars (pA)
- `t`: time (ms)

# State vector
`u = [V, w, DA]`

# Notes
Modulatory cell with slow dopamine release (tau ~ 200 ms).
Receives excitatory input from ON-bipolars.
"""
function da_model!(du, u, p, t)
    params, I_exc = p

    # Decompose state vector using tuple unpacking
    V, w, DA = u

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

    # Extract dopamine release parameters
    alpha_DA = params.alpha_DA
    V_DA_half = params.V_DA_half
    V_DA_slope = params.V_DA_slope
    tau_DA = params.tau_DA

    # Morris-Lecar activation functions
    m_inf = m_inf_ml(V, V1, V2)
    w_inf = w_inf_ml(V, V3, V4)
    tau_w = tau_w_ml(V, V3, V4)

    # Membrane currents
    I_L = g_L * (V - E_L)
    I_Ca = g_Ca * m_inf * (V - E_Ca)
    I_K = g_K * w * (V - E_K)

    # Derivatives
    dV = (-I_L - I_Ca - I_K + I_exc + params.I_app) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)

    # Dopamine release (very slow)
    T_inf = gate_inf(V, V_DA_half, V_DA_slope)
    dDA = (alpha_DA * T_inf - DA) / tau_DA

    # Assign derivatives
    du .= [dV, dw, dDA]

    return nothing
end
