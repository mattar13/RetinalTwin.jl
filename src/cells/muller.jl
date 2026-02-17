# ============================================================
# muller.jl - Müller glial cell K+ buffering + glutamate uptake
# Cleaned + stabilized:
#   (1) Resting V initialized near EK
#   (2) Kir4.1 inward rectification (prevents unphysiological outward Kir)
#   (3) K_o exchange driven by Kir current sign (instead of "uptake" heuristic)
#   (4) Small leak current added for numerical/physiological stability
#   (5) Clearer unit intent: K_o is "effective ECS K" (mM-like), requires scaling β_K
# ============================================================

const RT_F = 26.7  # mV at 37°C (RT/F for ln)

# ── 2. Initial Conditions ───────────────────────────────────

"""
    muller_state(params)

Return dark-adapted initial conditions for a Müller glial cell.

# Returns
- 4-element state vector [V_M, K_o_end, K_o_stalk, Glu_o]
"""
function muller_state(params)
    K_o_end   = params.K_o_rest
    K_o_stalk = params.K_o_rest
    # Initialize V near EK at rest (optionally slightly depolarized by leak)
    EK0 = nernst_K(params.K_o_rest, params.K_i)
    V0  = EK0
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

# ── 3. Auxiliary Functions ──────────────────────────────────

"""
    nernst_K(K_o, K_i)

Compute K+ Nernst potential (mV): E_K = (RT/F) * ln(K_o / K_i)
"""
@inline nernst_K(K_o::Real, K_i::Real) = RT_F * log(max(K_o, 1e-6) / max(K_i, 1e-6))

"""
    kir_rect(V, E_K; Vshift=15.0, k=6.0)

Phenomenological inward rectification factor for Kir4.1.
- ~1 when V is below E_K (inward regime)
- ~0 when V is above E_K (outward regime suppressed)

Vshift sets where rectification starts to shut off relative to E_K.
"""
@inline function kir_rect(V::Real, E_K::Real; Vshift::Real=15.0, k::Real=6.0)
    return 1.0 / (1.0 + exp((V - (E_K + Vshift)) / k))
end

"""
    muller_transmembrane_current(u, params)

Compute total transmembrane current (for ERG contribution).
Note: This includes Kir + leak (if desired, add other terms here).
"""
function muller_transmembrane_current(u, params::NamedTuple)
    V_M      = u[MULLER_V_INDEX]
    K_o_end  = u[MULLER_K_END_INDEX]
    K_o_stk  = u[MULLER_K_STALK_INDEX]

    E_K_end = nernst_K(K_o_end, params.K_i)
    E_K_stk = nernst_K(K_o_stk, params.K_i)

    r_end = kir_rect(V_M, E_K_end; Vshift=params.Kir_Vshift, k=params.Kir_k)
    r_stk = kir_rect(V_M, E_K_stk; Vshift=params.Kir_Vshift, k=params.Kir_k)

    I_Kir_end   = params.g_Kir_end   * r_end * (V_M - E_K_end)
    I_Kir_stalk = params.g_Kir_stalk * r_stk * (V_M - E_K_stk)
    I_L         = params.g_L * (V_M - params.E_L)

    return I_Kir_end + I_Kir_stalk + I_L
end

# ── 4. Mathematical Model ───────────────────────────────────

"""
    muller_model!(du, u, p, t)

Müller glial cell model with:
- Kir4.1-dominated membrane (inward-rectifying)
- Two ECS K+ pools (endfoot + stalk) with diffusion to rest
- Glutamate uptake via EAAT/GLAST (Michaelis-Menten)

# Arguments
- `du`: derivative vector (4 elements)
- `u`: state vector [V_M, K_o_end, K_o_stalk, Glu_o]
- `p`: tuple `(params, I_K_neural, Glu_release_total)` where:
  - `I_K_neural`: total nearby neuronal K+ current (pA; sign convention below)
  - `Glu_release_total`: glutamate source term into ECS (µM/ms or "a.u./ms")

# Sign conventions (recommended)
- Treat `I_K_neural` as a *source* that increases ECS K+ when neurons are active.
  If your upstream K current is outward-positive, you likely want:
      I_K_neural_source = max(I_K_neural, 0)
  and then scale into dK_o using alpha_K.
"""
function muller_model!(du, u, p, t)
    params, I_K_src_end, I_K_src_stalk, Glu_release_total = p

    V_M, K_o_end, K_o_stalk, Glu_o = u

    # --- Nernst potentials ---
    E_K_end   = nernst_K(K_o_end,   params.K_i)
    E_K_stalk = nernst_K(K_o_stalk, params.K_i)

    # --- Kir currents (rectifying) ---
    r_end = kir_rect(V_M, E_K_end;   Vshift=params.Kir_Vshift, k=params.Kir_k)
    r_stk = kir_rect(V_M, E_K_stalk; Vshift=params.Kir_Vshift, k=params.Kir_k)

    I_Kir_end   = params.g_Kir_end   * r_end * (V_M - E_K_end)
    I_Kir_stalk = params.g_Kir_stalk * r_stk * (V_M - E_K_stalk)

    # --- Leak current (small) ---
    I_L = params.g_L * (V_M - params.E_L)

    # --- K+ source from neurons ---
    # Use outward-positive K current as a source into ECS (clamp at 0)
    # Split neuronal source between pools
    K_src_end   = params.alpha_K * params.frac_end   * I_K_src_end
    K_src_stalk = params.alpha_K * params.frac_stalk * I_K_src_stalk

    # --- K+ exchange via Kir (glia buffers ECS K) ---
    # Inward Kir current is negative. If Kir is inward (I_Kir < 0),
    # glia takes up K+ from ECS => K_o decreases.
    # Use (-I_Kir) as "uptake strength" and scale by beta_K.
    K_buf_end   = params.beta_K * (-I_Kir_end)
    K_buf_stalk = params.beta_K * (-I_Kir_stalk)

    # --- Relaxation back to rest (diffusion / clearance) ---
    K_relax_end   = (K_o_end   - params.K_o_rest) / params.tau_K_diffusion
    K_relax_stalk = (K_o_stalk - params.K_o_rest) / params.tau_K_diffusion

    # --- K_o dynamics ---
    dK_o_end   = K_src_end   - K_buf_end   - K_relax_end
    dK_o_stalk = K_src_stalk - K_buf_stalk - K_relax_stalk

    # Keep K_o nonnegative (softly). If you prefer pure dynamics, remove these max().
    # This is not "S_min/S_max" clamping; it's preventing log-domain failure.
    if K_o_end + dK_o_end * params.dt_guard < 1e-6
        dK_o_end = (1e-6 - K_o_end) / max(params.dt_guard, 1e-3)
    end
    if K_o_stalk + dK_o_stalk * params.dt_guard < 1e-6
        dK_o_stalk = (1e-6 - K_o_stalk) / max(params.dt_guard, 1e-3)
    end

    # --- Glutamate uptake (EAAT/GLAST) ---
    Glu_o_clamped = max(Glu_o, 0.0)
    J_uptake = params.V_max_EAAT * Glu_o_clamped / (params.K_m_EAAT + Glu_o_clamped + eps())
    I_EAAT = -params.g_EAAT * J_uptake

    # --- Voltage ---
    dV_M = (-(I_Kir_end + I_Kir_stalk + I_L + I_EAAT) + params.I_app) / params.C_m
    
    # --- Glu dynamics ---
    dGlu_o = Glu_release_total - J_uptake

    du .= (dV_M, dK_o_end, dK_o_stalk, dGlu_o)
    return nothing
end
