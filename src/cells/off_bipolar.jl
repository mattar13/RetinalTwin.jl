# ============================================================
# off_bipolar.jl - OFF-Bipolar cell dynamics
# ============================================================

# ── 2. Initial Conditions ───────────────────────────────────

"""
    offbc_state(params)

Return dark-adapted initial conditions for an OFF bipolar cell.

# Arguments
- `params`: named tuple from `default_off_bc_params()`

# Returns
- 7-element state vector [V, n, h, c, A, D, G]
"""
function offbc_state(params)
    V0 = -60.0
    n0 = gate_inf(V0, params.Vn_half, params.kn_slope)
    h0 = gate_inf(V0, params.Vh_half, params.kh_slope)
    c0 = 0.0
    A0 = 0.0
    D0 = 0.0
    G0 = 0.0
    return [V0, n0, h0, c0, A0, D0, G0]
end

# ── 3. Auxiliary Functions ──────────────────────────────────

#This function is already defined in on_bipolar.jl
# @inline function hill(x, K, n)
#     # bounded [0,1], handles x>=0
#     xn = x^n
#     return xn / (K^n + xn + eps())
# end
#So is this one
# @inline function gate_inf(V, Vhalf, k)
#     # logistic with slope k (mV). k can be negative (e.g. Ih)
#     return 1.0 / (1.0 + exp(-(V - Vhalf) / k))
# end

"""
OFF ionotropic receptor activation function
"""
@inline function A_inf(glu, K, n)
    # Increasing Hill: more glutamate -> more activation (OFF pathway)
    return hill(max(glu, 0.0), K, n)
end

"""
OFF ionotropic receptor desensitization function
"""
@inline function D_inf(glu, K, n)
    # Fraction available (NOT desensitized).
    # More glutamate -> more desensitization -> less availability.
    # 1 / (1 + (glu/K)^n)
    return 1.0 / (1.0 + (max(glu, 0.0) / K)^n)
end
# ── 4. Mathematical Model ───────────────────────────────────

"""
    off_bipolar_model!(du, u, p, t)

Morris-Lecar OFF bipolar cell model with ionotropic glutamate receptor.

# Arguments
- `du`: derivative vector (4 elements)
- `u`: state vector (4 elements)
- `p`: tuple `(params, glu_mean, I_inh, I_mod)` where:
  - `params`: named tuple from `default_off_bc_params()`
  - `glu_mean`: mean glutamate concentration from photoreceptors
  - `I_inh`: inhibitory synaptic current (pA)
  - `I_mod`: modulatory current (pA)
- `t`: time (ms)

# State vector
`u = [V, n, h, c, A, D, Glu_release]`

# Notes
Ionotropic synapse: depolarizes when glutamate is high (dark),
hyperpolarizes when Glu drops (light).
"""
function off_bipolar_model!(du, u, p, t)
    params, glu_received = p

    # Decompose state vector using tuple unpacking
    V, n, h, c, A, D, G = u

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
    dG = (params.a_Release * R_inf(c, params.K_Release, params.n_Release) - G) / params.tau_Release

    du .= (dV, dn, dh, dc, dA, dD, dG)
    return nothing
end
