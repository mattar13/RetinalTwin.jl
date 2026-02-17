# ============================================================
# a2_amacrine.jl - A2 (AII) Amacrine cell dynamics
# ============================================================

# ── 2. Initial Conditions ───────────────────────────────────

"""
    a2_dark_state(params)

Return dark-adapted initial conditions for an A2 amacrine cell.

# Arguments
- `params`: named tuple from `default_a2_params()`

# Returns
- 7-element state vector [V, n, h, c, A, D, Y]
"""
function a2_state(params)
    V0 = -60.0
    n0 = gate_inf(V0, params.Vn_half, params.kn_slope)
    h0 = gate_inf(V0, params.Vh_half, params.kh_slope)
    c0 = 0.0
    A0 = 0.0
    D0 = 0.0
    Y0 = 0.0
    return [V0, n0, h0, c0, A0, D0, Y0]
end

# ── 3. Auxiliary Functions ──────────────────────────────────

#Defined in off_bipolar.jl

# """
# OFF ionotropic receptor activation function
# """
# @inline function A_inf(glu, K, n)
#     # Increasing Hill: more glutamate -> more activation (OFF pathway)
#     return hill(max(glu, 0.0), K, n)
# end

# """
# OFF ionotropic receptor desensitization function
# """
# @inline function D_inf(glu, K, n)
#     # Fraction available (NOT desensitized).
#     # More glutamate -> more desensitization -> less availability.
#     # 1 / (1 + (glu/K)^n)
#     return 1.0 / (1.0 + (max(glu, 0.0) / K)^n)
# end
# ── 4. Mathematical Model ───────────────────────────────────

"""
    a2_model!(du, u, p, t)

Morris-Lecar A2 (AII) amacrine cell model.

# Arguments
- `du`: derivative vector (7 elements)
- `u`: state vector (7 elements)
- `p`: tuple `(params, I_exc, I_inh, I_mod)` where:
  - `params`: named tuple from `default_a2_params()`
  - `I_exc`: excitatory synaptic current from bipolars (pA)
  - `I_inh`: inhibitory synaptic current (pA)
  - `I_mod`: modulatory current (pA)
- `t`: time (ms)

# State vector
`u = [V, n, h, c, A, D, Y]`

# Notes
Critical for oscillatory potential generation. Fast ML dynamics
(low C_m, high phi) enable high-frequency oscillations.
"""
function a2_model!(du, u, p, t)
    params, glu_received = p

    # Decompose state vector using tuple unpacking
    V, n, h, c, A, D, Y = u

    # -------- OFF ionotropic receptor (AMPA/KAR-like) --------
    A_INF = A_inf(glu_received, params.K_a, params.n_a)
    dA = (params.a_a * A_INF - A) / params.tau_A

    # -------- desensitization --------
    # Desensitization (optional but recommended for realistic OFF kinetics)
    # If you don't want desensitization, set D=1 always by:
    #   D = 1.0, dD = 0.0, or set tau_des huge and tau_rec small.
    D_INF = D_inf(glu_received, params.K_d, params.n_d)
    dD = (params.a_d * D_INF - D) / params.tau_d

    #-------- effective open probability --------
    open_iGluR = A * D  # effective open probability

    # Morris-Lecar activation functions
    n_inf = gate_inf(V, params.Vn_half, params.kn_slope)
    dn = (n_inf - n) / params.tau_n

    h_inf = gate_inf(V, params.Vh_half, params.kh_slope) # kh_slope < 0 for Ih
    dh = (h_inf - h) / params.tau_h

    m_inf = gate_inf(V, params.Vm_half, params.km_slope) # CaL activation (instant-ish)
    # if you want dynamic m, swap to a state variable; here we keep it simple

    # -------- currents --------
    I_L     = params.g_L * (V - params.E_L)
    I_Kv    = params.g_Kv * n * (V - params.E_K)
    I_h     = params.g_h * h * (V - params.E_h)
    I_CaL   = params.g_CaL * m_inf * (V - params.E_Ca)
    # OFF synaptic current (nonselective cation, reversal ~0 mV)
    I_iGluR = params.g_iGluR * open_iGluR * (V - params.E_iGluR)

        #-------- Ca pool --------
    # driven by inward Ca current only (when I_CaL is negative)
    Ca_in = max(-I_CaL, 0.0)
    dc = (-c / params.tau_c) + params.k_c * Ca_in

    # KCa activation from Ca pool
    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    I_KCa = params.g_KCa * a_c * (V - params.E_K)

    # -------- voltage derivative --------
    dV = (-I_L - I_iGluR - I_Kv - I_h - I_CaL - I_KCa + params.I_app) / params.C_m

    # -------- output glutamate release proxy (Ca-driven) --------
    dY = (params.a_Release * R_inf(c, params.K_Release, params.n_Release) - Y) / params.tau_Release

    du .= (dV, dn, dh, dc, dA, dD, dY)
    return nothing
end
