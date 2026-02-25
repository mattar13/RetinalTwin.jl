# ============================================================
# gaba_amacrine.jl - GABAergic Amacrine cell dynamics
# ============================================================

"""
    gaba_dark_state(params)

Return dark-adapted initial conditions for a GABAergic amacrine cell.

# Arguments
- `params`: named tuple from `load_all_params().GABA`

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

# gaba_model!(du, u, p, t)
# Morris-Lecar GABAergic amacrine cell model.
# State: u = [V, w, GABA]
"""
    I_gaba_amacrine(V, params)

Approximate total GABA amacrine transmembrane current at voltage `V` using
steady-state Morris-Lecar gates.
"""
function I_gaba_amacrine(V, params)
    m_inf = m_inf_ml(V, params.V1, params.V2)
    w_inf = w_inf_ml(V, params.V3, params.V4)

    I_L = params.g_L * (V - params.E_L)
    I_Ca = params.g_Ca * m_inf * (V - params.E_Ca)
    I_K = params.g_K * w_inf * (V - params.E_K)

    return I_L + I_Ca + I_K
end

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
    dV = (-I_L - I_Ca - I_K + I_exc + I_inh + I_mod + params.I_app) / C_m
    dw = phi * (w_inf - w) / max(tau_w, 0.1)

    # GABA release
    T_inf = gate_inf(V, V_GABA_half, V_GABA_slope)
    dGABA = (alpha_GABA * T_inf - GABA) / tau_GABA

    # Assign derivatives
    du .= [dV, dw, dGABA]

    return nothing
end
