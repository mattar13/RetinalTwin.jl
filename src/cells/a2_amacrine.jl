# ============================================================
# a2_amacrine.jl — A2 (AII) Amacrine cell, glycinergic
# State vector: [V, w, Gly] (3 variables per cell)
# Spec §3.5 — Critical for oscillatory potential generation
# ============================================================

"""
    update_a2!(du, u, ml::MLParams, I_exc, I_inh, I_mod)

Compute derivatives for one A2 amacrine cell.
`u = [V, w, Gly]`.
Fast ML dynamics (low C_m, high phi) for OP oscillation generation.
"""
function update_a2!(du, u, ml::MLParams,
                    I_exc::Real, I_inh::Real, I_mod::Real)
    V   = u[1]
    w   = u[2]
    Gly = u[3]

    # ML activation functions
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))

    # Membrane potential
    du[1] = (-ml.g_L * (V - ml.E_L) -
              ml.g_Ca * m_inf * (V - ml.E_Ca) -
              ml.g_K * w * (V - ml.E_K) +
              I_exc + I_inh + I_mod) / ml.C_m

    # Recovery variable
    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)

    # Glycine release (fast, critical for OP frequency)
    T_inf = 1.0 / (1.0 + exp(-(V - ml.release.V_half) / ml.release.V_slope))
    du[3] = (ml.release.alpha * T_inf - Gly) / ml.release.tau

    return nothing
end
