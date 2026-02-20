# ============================================================
# off_bipolar.jl - OFF-Bipolar cell dynamics
# ============================================================

"""
    off_bipolar_state(params)

Return dark-adapted initial conditions for an OFF bipolar cell.
"""
function off_bipolar_state(params)
    V0 = -60.0
    n0 = gate_inf(V0, params.Vn_half, params.kn_slope)
    h0 = gate_inf(V0, params.Vh_half, params.kh_slope)
    c0 = 0.0
    A0 = 0.0
    D0 = 0.0
    G0 = 0.0
    return [V0, n0, h0, c0, A0, D0, G0]
end

const OFFBC_IC_MAP = (
    V = 1,
    n = 2,
    h = 3,
    c = 4,
    A = 5,
    D = 6,
    Glu = 7
)

n_OFFBC_STATES = length(OFFBC_IC_MAP)

"""
    off_bipolar_model!(du, u, p, t)

Morris-Lecar OFF bipolar cell model with ionotropic glutamate receptor.
"""
function off_bipolar_model!(du, u, p, t)
    params, glu_in, w_glu_in = p
    V, n, h, c, A, D, G = u

    A_INF = spatial_synaptic(glu_in, w_glu_in, params, :hill, :K_a, :n_a)
    dA = (params.a_a * A_INF - A) / params.tau_A

    D_INF = spatial_synaptic(glu_in, w_glu_in, params, :inv_hill, :K_d, :n_d)
    dD = (params.a_d * D_INF - D) / params.tau_d

    open_iGluR = A * D

    n_inf = gate_inf(V, params.Vn_half, params.kn_slope)
    dn = (n_inf - n) / params.tau_n

    h_inf = gate_inf(V, params.Vh_half, params.kh_slope)
    dh = (h_inf - h) / params.tau_h

    m_inf = gate_inf(V, params.Vm_half, params.km_slope)

    I_L = params.g_L * (V - params.E_L)
    I_Kv = params.g_Kv * n * (V - params.E_K)
    I_h = params.g_h * h * (V - params.E_h)
    I_CaL = params.g_CaL * m_inf * (V - params.E_Ca)
    I_iGluR = params.g_iGluR * open_iGluR * (V - params.E_iGluR)

    Ca_in = max(-I_CaL, 0.0)
    dc = (-c / params.tau_c) + params.k_c * Ca_in

    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    I_KCa = params.g_KCa * a_c * (V - params.E_K)

    dV = (-I_L - I_iGluR - I_Kv - I_h - I_CaL - I_KCa + params.I_app) / params.C_m
    dG = (params.a_Release * R_inf(c, params.K_Release, params.n_Release) - G) / params.tau_Release

    du .= (dV, dn, dh, dc, dA, dD, dG)
    return nothing
end

function off_bipolar_K_efflux(u, params)
    V = u[OFFBC_IC_MAP.V]
    n = u[OFFBC_IC_MAP.n]
    EK = params.E_K
    IKv = params.g_Kv * n * (V - EK)
    return IKv
end
