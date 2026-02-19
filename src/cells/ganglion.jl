# ============================================================
# ganglion.jl - Ganglion cell dynamics
# ============================================================

"""
    ganglion_state(params)

Return dark-adapted initial conditions for a ganglion cell.
"""
function ganglion_state(params)
    V0 = -60.0
    αm, βm = alpha_beta_m(V0)
    αh, βh = alpha_beta_h(V0)
    αn, βn = alpha_beta_n(V0)
    m0 = αm / (αm + βm + eps())
    h0 = αh / (αh + βh + eps())
    n0 = αn / (αn + βn + eps())
    sE0 = 0.0
    sI0 = 0.0
    return [V0, m0, h0, n0, sE0, sI0]
end

const GC_IC_MAP = (
    V = 1,
    m = 2,
    h = 3,
    n = 4,
    sE = 5,
    sI = 6
)

n_GC_STATES = length(GC_IC_MAP)

"""
    ganglion_model!(du, u, p, t)

Hodgkin-Huxley style ganglion cell model.
"""
function ganglion_model!(du, u, p, t)
    params, glu_in, gly_in = p
    V, m, h, n, sE, sI = u

    sE_inf = hill(glu_in, params.K_preE, params.n_preE)
    sI_inf = hill(gly_in, params.K_preI, params.n_preI)

    dsE = (params.a_preE * sE_inf - sE) / params.tau_E
    dsI = (params.a_preI * sI_inf - sI) / params.tau_I

    I_L = params.g_L * (V - params.E_L)
    I_Na = params.g_Na * (m^3) * h * (V - params.E_Na)
    I_K = params.g_K * (n^4) * (V - params.E_K)
    I_E = params.g_E * sE * (V - params.E_E)
    I_I = params.g_I * sI * (V - params.E_I)

    dV = (-I_L - I_Na - I_K - I_E - I_I + params.I_app) / params.C_m

    αm, βm = alpha_beta_m(V)
    αh, βh = alpha_beta_h(V)
    αn, βn = alpha_beta_n(V)

    dm = αm * (1 - m) - βm * m
    dh = αh * (1 - h) - βh * h
    dn = αn * (1 - n) - βn * n

    du .= (dV, dm, dh, dn, dsE, dsI)
    return nothing
end

function ganglion_K_efflux(u, params)
    V = u[GC_IC_MAP.V]
    n = u[GC_IC_MAP.n]
    EK = params.E_K
    IKv = params.g_K * (n^4) * (V - EK)
    return IKv
end
