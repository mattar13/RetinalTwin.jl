# ============================================================
# da_amacrine.jl — Dopaminergic amacrine cell
# State vector: [V, w, DA] (3 variables per cell)
# Spec §3.7 — Modulatory, slow dopamine release
# ============================================================

"""
    update_da_amacrine!(du, u, ml::MLParams, I_exc)

Compute derivatives for one dopaminergic amacrine cell.
`u = [V, w, DA]`.
Receives excitatory input from ON-bipolars. Releases dopamine slowly (tau ~ 200 ms).
"""
function update_da_amacrine!(du, u, ml::MLParams, I_exc::Real)
    V  = u[1]
    w  = u[2]
    DA = u[3]

    # ML activation functions
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))

    # Membrane potential (only excitatory input in Phase 1)
    du[1] = (-ml.g_L * (V - ml.E_L) -
              ml.g_Ca * m_inf * (V - ml.E_Ca) -
              ml.g_K * w * (V - ml.E_K) +
              I_exc) / ml.C_m

    # Recovery variable
    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)

    # Dopamine release (very slow)
    T_inf = 1.0 / (1.0 + exp(-(V - ml.release.V_half) / ml.release.V_slope))
    du[3] = (ml.release.alpha * T_inf - DA) / ml.release.tau

    return nothing
end
