# ============================================================
# muller.jl - Muller glial cell K+ buffering + glutamate uptake
# ============================================================

"""
    muller_state(params)

Return dark-adapted initial conditions for a Muller glial cell.
"""
function muller_state(params)
    K_o_end = params.K_o_rest
    K_o_stalk = params.K_o_rest
    EK0 = nernst_K(params.K_o_rest, params.K_i)
    V0 = EK0
    Glu_o = 0.0
    return [V0, K_o_end, K_o_stalk, Glu_o]
end

const MG_IC_MAP = (
    V = 1,
    K_o_end = 2,
    K_o_stalk = 3,
    Glu_o = 4
)

n_MG_STATES = length(MG_IC_MAP)

"""
    I_muller(V, params)

Approximate total Muller transmembrane current at voltage `V` using resting K+
for Kir reversal estimates.
"""
function I_muller(V, params)
    E_K = nernst_K(params.K_o_rest, params.K_i)
    r = kir_rect(V, E_K; Vshift=params.Kir_Vshift, k=params.Kir_k)

    I_Kir_end = params.g_Kir_end * r * (V - E_K)
    I_Kir_stalk = params.g_Kir_stalk * r * (V - E_K)
    I_L = params.g_L * (V - params.E_L)

    return I_Kir_end + I_Kir_stalk + I_L
end

"""
    muller_transmembrane_current(u, params)

Compute total transmembrane current (for ERG contribution).
"""
function muller_transmembrane_current(u, params)
    V_M = u[MG_IC_MAP.V]
    K_o_end = u[MG_IC_MAP.K_o_end]
    K_o_stk = u[MG_IC_MAP.K_o_stalk]

    E_K_end = nernst_K(K_o_end, params.K_i)
    E_K_stk = nernst_K(K_o_stk, params.K_i)

    r_end = kir_rect(V_M, E_K_end; Vshift=params.Kir_Vshift, k=params.Kir_k)
    r_stk = kir_rect(V_M, E_K_stk; Vshift=params.Kir_Vshift, k=params.Kir_k)

    I_Kir_end = params.g_Kir_end * r_end * (V_M - E_K_end)
    I_Kir_stalk = params.g_Kir_stalk * r_stk * (V_M - E_K_stk)
    I_L = params.g_L * (V_M - params.E_L)

    return I_Kir_end + I_Kir_stalk + I_L
end

"""
    muller_model!(du, u, p, t)

Muller glial cell model.
"""
function muller_model!(du, u, p, t)
    params, I_K_src_end, I_K_src_stalk, Glu_release_total = p
    V_M, K_o_end, K_o_stalk, Glu_o = u

    E_K_end = nernst_K(K_o_end, params.K_i)
    E_K_stalk = nernst_K(K_o_stalk, params.K_i)

    r_end = kir_rect(V_M, E_K_end; Vshift=params.Kir_Vshift, k=params.Kir_k)
    r_stk = kir_rect(V_M, E_K_stalk; Vshift=params.Kir_Vshift, k=params.Kir_k)

    I_Kir_end = params.g_Kir_end * r_end * (V_M - E_K_end)
    I_Kir_stalk = params.g_Kir_stalk * r_stk * (V_M - E_K_stalk)
    I_L = params.g_L * (V_M - params.E_L)

    K_src_end = params.alpha_K * params.frac_end * I_K_src_end
    K_src_stalk = params.alpha_K * params.frac_stalk * I_K_src_stalk

    K_buf_end = params.beta_K * (-I_Kir_end)
    K_buf_stalk = params.beta_K * (-I_Kir_stalk)

    K_relax_end = (K_o_end - params.K_o_rest) / params.tau_K_diffusion
    K_relax_stalk = (K_o_stalk - params.K_o_rest) / params.tau_K_diffusion

    dK_o_end = K_src_end - K_buf_end - K_relax_end
    dK_o_stalk = K_src_stalk - K_buf_stalk - K_relax_stalk

    if K_o_end + dK_o_end * params.dt_guard < 1e-6
        dK_o_end = (1e-6 - K_o_end) / max(params.dt_guard, 1e-3)
    end
    if K_o_stalk + dK_o_stalk * params.dt_guard < 1e-6
        dK_o_stalk = (1e-6 - K_o_stalk) / max(params.dt_guard, 1e-3)
    end

    Glu_o_clamped = max(Glu_o, 0.0)
    J_uptake = params.V_max_EAAT * hill(Glu_o_clamped, params.K_m_EAAT, params.n_EAAT)
    I_EAAT = -params.g_EAAT * J_uptake

    dV_M = (-(I_Kir_end + I_Kir_stalk + I_L + I_EAAT) + params.I_app) / params.C_m
    dGlu_o = Glu_release_total - J_uptake

    du .= (dV_M, dK_o_end, dK_o_stalk, dGlu_o)
    return nothing
end
