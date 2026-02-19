# ============================================================
# horizontal.jl - Horizontal cell dynamics
# ============================================================

"""
    horizontal_state(params)

Return dark-adapted initial conditions for a horizontal cell.
"""
function horizontal_state(params)
    V0 = -60.0
    c0 = 0.0
    I0 = 0.0
    return [V0, c0, I0]
end

hc_dark_state(params) = horizontal_state(params)

const HC_IC_MAP = (
    V = 1,
    c = 2,
    I = 3
)

n_HC_STATES = length(HC_IC_MAP)

"""
    horizontal_model!(du, u, p, t)

Horizontal cell model with static/single-state AMPA-KA-like excitation,
Kir and BK currents, and Ca-driven GABA-release proxy.
"""
function horizontal_model!(du, u, p, t)
    params, I_ext, I_gap, glu_mean = p
    V, c, I = u

    # Stage-1 robust glutamatergic drive: static Hill activation.
    s_inf = hill(max(glu_mean, 0.0), params.K_Glu, params.n_Glu)

    I_L = params.g_L * (V - params.E_L)
    I_exc = params.g_exc * s_inf * (V - params.E_exc)

    mCa = gate_inf(V, params.Vm_half, params.km_slope)
    I_CaL = params.g_CaL * mCa * (V - params.E_Ca)

    r_kir = kir_rect(V, params.E_Kir; Vshift=params.Kir_Vshift, k=params.Kir_k)
    I_Kir = params.g_Kir * r_kir * (V - params.E_Kir)

    m = mBK_inf(V, c; Vhalf0=params.Vhalf0_BK, k=params.k_BK, s=params.s_BK, Caref=params.Caref_BK)
    I_BK = params.gBK * m * (V - params.E_K)

    Ca_in = max(-I_CaL, 0.0)
    dc = (-c / max(params.tau_c, 1e-6)) + params.k_c * Ca_in

    I_inf = params.a_Release * R_inf(c, params.K_Release, params.n_Release)
    dI = (I_inf - I) / max(params.tau_Release, 1e-6)

    dV = (-I_L - I_exc - I_CaL - I_Kir - I_BK + I_ext + I_gap + params.I_app) / params.C_m

    du .= (dV, dc, dI)
    return nothing
end
