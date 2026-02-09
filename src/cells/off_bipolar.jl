# ============================================================
# off_bipolar.jl — OFF-Bipolar cell with ionotropic glutamate receptor
# State vector: [V, w, s_Glu, Glu_release] (4 variables per cell)
# Spec §3.4
# ============================================================

"""
    update_off_bipolar!(du, u, ml::MLParams, glu_pre, I_inh, I_mod)

Compute derivatives for one OFF-bipolar cell.
`u = [V, w, s_Glu, Glu_release]`.
Ionotropic: depolarizes when glutamate is high (dark), hyperpolarizes when Glu drops (light).
"""
function update_off_bipolar!(du, u, ml::MLParams,
                             glu_pre::Real, I_inh::Real, I_mod::Real)
    V       = u[1]
    w       = u[2]
    s_Glu   = u[3]
    Glu_rel = u[4]

    # Ionotropic glutamate gating (fast, tau ~ 3 ms)
    tau_iGluR = 3.0
    du[3] = (glu_pre - s_Glu) / tau_iGluR

    # Excitatory synaptic current from photoreceptor glutamate
    g_iGluR = 4.0  # nS (from connection table)
    E_E = 0.0      # mV
    I_Glu = g_iGluR * s_Glu * (V - E_E)

    # ML dynamics
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))

    du[1] = (-ml.g_L * (V - ml.E_L) -
              ml.g_Ca * m_inf * (V - ml.E_Ca) -
              ml.g_K * w * (V - ml.E_K) +
              I_Glu + I_inh + I_mod) / ml.C_m

    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)

    # Glutamate release (graded)
    R_glu = 1.0 / (1.0 + exp(-(V - ml.release.V_half) / ml.release.V_slope))
    du[4] = (ml.release.alpha * R_glu - Glu_rel) / ml.release.tau

    return nothing
end
