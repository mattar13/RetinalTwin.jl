# ============================================================
# horizontal.jl — Horizontal cell dynamics
# State vector: [V, w, s_Glu] (3 variables per cell)
# Spec §3.2
# ============================================================

"""
    update_horizontal!(du, u, ml::MLParams, I_exc, I_gap)

Compute derivatives for one horizontal cell.
`u = [V, w, s_Glu]`.
`I_exc` = pre-computed excitatory synaptic current from photoreceptors.
`I_gap` = gap junction current from neighboring HCs.
"""
function update_horizontal!(du, u, ml::MLParams, I_exc::Real, I_gap::Real)
    V     = u[1]
    w     = u[2]
    s_Glu = u[3]

    # ML activation functions
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))

    # Membrane potential
    du[1] = (-ml.g_L * (V - ml.E_L) -
              ml.g_Ca * m_inf * (V - ml.E_Ca) -
              ml.g_K * w * (V - ml.E_K) +
              I_exc + I_gap) / ml.C_m

    # Recovery variable
    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)

    # s_Glu tracks separately (updated by caller via synapse framework)
    # Here we just zero it; the ODE RHS handles s_Glu tracking
    du[3] = 0.0

    return nothing
end

"""
    hc_feedback(V_hc; g_FB=1.0, V_FB_half=-50.0, V_FB_slope=5.0)

Compute HC feedback signal to photoreceptors.
Returns a feedback current magnitude.
"""
function hc_feedback(V_hc::Real; g_FB::Real=1.0,
                     V_FB_half::Real=-50.0, V_FB_slope::Real=5.0)
    return g_FB / (1.0 + exp(-(V_hc - V_FB_half) / V_FB_slope))
end
