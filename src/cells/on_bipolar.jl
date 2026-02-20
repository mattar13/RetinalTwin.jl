# ============================================================
# on_bipolar.jl - ON-Bipolar cell dynamics
# ============================================================

"""
    on_bipolar_state(params)

Return dark-adapted initial conditions for an ON bipolar cell.
"""
function on_bipolar_state(params)
    V0 = -60.0
    n0 = gate_inf(V0, params.Vn_half, params.kn_slope)
    h0 = gate_inf(V0, params.Vh_half, params.kh_slope)
    c0 = 0.0
    S0 = 0.0
    Glu0 = 0.0
    return [V0, n0, h0, c0, S0, Glu0]
end

const ONBC_IC_MAP = (
    V = 1,
    n = 2,
    h = 3,
    c = 4,
    S = 5,
    Glu = 6
)

n_ONBC_STATES = length(ONBC_IC_MAP)

"""
    on_bipolar_model!(du, u, p, t)

Morris-Lecar ON bipolar cell model with mGluR6 sign inversion.
"""
function on_bipolar_model!(du, u, p, t)
    params, glu_in, w_glu_in = p
    V, n, h, c, S, G = u

    S_INF = spatial_synaptic(glu_in, w_glu_in, params, :inv_hill, :K_Glu, :n_Glu)
    dS = (params.a_S * S_INF - S) / params.tau_S

    n_inf = gate_inf(V, params.Vn_half, params.kn_slope)
    dn = (n_inf - n) / params.tau_n

    h_inf = gate_inf(V, params.Vh_half, params.kh_slope)
    dh = (h_inf - h) / params.tau_h

    m_inf = gate_inf(V, params.Vm_half, params.km_slope)

    I_L = params.g_L * (V - params.E_L)
    I_TRPM1 = params.g_TRPM1 * S * (V - params.E_TRPM1)
    I_Kv = params.g_Kv * n * (V - params.E_K)
    I_h = params.g_h * h * (V - params.E_h)
    I_CaL = params.g_CaL * m_inf * (V - params.E_Ca)

    Ca_in = max(-I_CaL, 0.0)
    dc = (-c / params.tau_c) + params.k_c * Ca_in

    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    I_KCa = params.g_KCa * a_c * (V - params.E_K)

    dV = (-I_L - I_TRPM1 - I_Kv - I_h - I_CaL - I_KCa + params.I_app) / params.C_m
    dG = (params.a_Release * R_inf(c, params.K_Release, params.n_Release) - G) / params.tau_Release

    du .= (dV, dn, dh, dc, dS, dG)
    return nothing
end

function on_bipolar_K_efflux(u, params)
    V = u[ONBC_IC_MAP.V]
    n = u[ONBC_IC_MAP.n]
    EK = params.E_K
    IKv = params.g_Kv * n * (V - EK)
    return IKv
end
