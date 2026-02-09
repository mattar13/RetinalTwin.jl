# ============================================================
# ganglion.jl — Ganglion cell dynamics
# State vector: [V, w] (2 variables per cell)
# Spec §3.8 — Output neuron, action potential generation
# ============================================================

"""
    update_ganglion!(du, u, ml::MLParams, I_exc, I_inh)

Compute derivatives for one ganglion cell.
`u = [V, w]`.
Receives excitatory input from ON/OFF bipolars, inhibitory from amacrines.
"""
function update_ganglion!(du, u, ml::MLParams, I_exc::Real, I_inh::Real)
    V = u[1]
    w = u[2]

    # ML activation functions
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))

    # Membrane potential
    du[1] = (-ml.g_L * (V - ml.E_L) -
              ml.g_Ca * m_inf * (V - ml.E_Ca) -
              ml.g_K * w * (V - ml.E_K) +
              I_exc + I_inh) / ml.C_m

    # Recovery variable
    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)

    return nothing
end
