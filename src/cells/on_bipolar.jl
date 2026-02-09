# ============================================================
# on_bipolar.jl — ON-Bipolar cell with mGluR6 sign inversion
# State vector: [V, w, S_mGluR6, Glu_release] (4 variables per cell)
# Spec §3.3
# ============================================================

"""
    update_on_bipolar!(du, u, ml::MLParams, mg::mGluR6Params,
                       glu_pre, I_inh, I_mod)

Compute derivatives for one ON-bipolar cell.
`u = [V, w, S_mGluR6, Glu_release]`.
The mGluR6 synapse inverts the glutamate signal: high Glu → cell hyperpolarized.
"""
function update_on_bipolar!(du, u, ml::MLParams, mg::mGluR6Params,
                            glu_pre::Real, I_inh::Real, I_mod::Real)
    V       = u[1]
    w       = u[2]
    S       = u[3]
    Glu_rel = u[4]

    # mGluR6 cascade: tracks glutamate with metabotropic kinetics
    du[3] = (mg.alpha_mGluR6 * glu_pre - S) / mg.tau_mGluR6

    # TRPM1 conductance (sign-inverted: low S → high conductance → depolarization)
    g_TRPM1 = mg.g_TRPM1_max * (1.0 - clamp(S, 0.0, 1.0))
    I_TRPM1 = g_TRPM1 * (V - mg.E_TRPM1)

    # ML dynamics
    m_inf = 0.5 * (1.0 + tanh((V - ml.V1) / ml.V2))
    w_inf = 0.5 * (1.0 + tanh((V - ml.V3) / ml.V4))
    tau_w = 1.0 / cosh((V - ml.V3) / (2.0 * ml.V4))

    # Dopamine gain modulation applied to TRPM1
    du[1] = (-ml.g_L * (V - ml.E_L) -
              ml.g_Ca * m_inf * (V - ml.E_Ca) -
              ml.g_K * w * (V - ml.E_K) -
              I_TRPM1 + I_inh + I_mod) / ml.C_m

    du[2] = ml.phi * (w_inf - w) / max(tau_w, 0.1)

    # Glutamate release (graded)
    R_glu = 1.0 / (1.0 + exp(-(V - ml.release.V_half) / ml.release.V_slope))
    du[4] = (ml.release.alpha * R_glu - Glu_rel) / ml.release.tau

    return nothing
end

# Clamp helper
clamp(x, lo, hi) = min(max(x, lo), hi)
