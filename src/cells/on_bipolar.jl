# ============================================================
# on_bipolar.jl - ON-Bipolar cell dynamics
# ============================================================

# ── 2. Initial Conditions ───────────────────────────────────

"""
    on_bipolar_dark_state(params)

Return dark-adapted initial conditions for an ON bipolar cell.

# Arguments
- `params`: named tuple from `default_on_bc_params()`

# Returns
- 6-element state vector [V, w, S_mGluR6, Glu_release, Ca, Glu]
"""
function onbc_state(params)
    V0 = -60.0
    n0 = gate_inf(V0, params.Vn_half, params.kn_slope)
    h0 = gate_inf(V0, params.Vh_half, params.kh_slope)
    c0 = 0.0
    S0 = 0.0
    Glu0 = 0.0
    return [V0, n0, h0, c0, S0, Glu0]
end

# ── 3. Auxiliary Functions ──────────────────────────────────

@inline σ(x) = 1.0 / (1.0 + exp(-x))

@inline function gate_inf(V, Vhalf, k)
    # logistic with slope k (mV). k can be negative (e.g. Ih)
    return 1.0 / (1.0 + exp(-(V - Vhalf) / k))
end

@inline function hill(x, K, n)
    # bounded [0,1], handles x>=0
    xn = x^n
    return xn / (K^n + xn + eps())
end

"""
mGluR6 activation function
"""
@inline S_inf(glu_received, K_Glu, n_Glu) = 1 / (1 + (glu_received / K_Glu)^n_Glu)

"""
Release function for glutamate release from the ON bipolar cell.
"""
@inline R_inf(Ca, K_Release, n_Release) = (Ca^n_Release) / (K_Release^n_Release + Ca^n_Release + eps())
# ── 4. Mathematical Model ───────────────────────────────────

"""
    on_bipolar_model!(du, u, p, t)

Morris-Lecar ON bipolar cell model with mGluR6 sign inversion.

# Arguments
- `du`: derivative vector (6 elements)
- `u`: state vector (6 elements)
- `p`: tuple `(params, glu_received)` where:
  - `params`: named tuple from `default_on_bc_params()` (includes mGluR6 parameters)
  - `glu_received`: glutamate concentration from photoreceptors (µM)
- `t`: time (ms)

# State vector
`u = [V, n, h, c, S_mGluR6, Glu_release]`

# Notes
The mGluR6 synapse inverts the glutamate signal: high Glu → cell hyperpolarized.
TRPM1 conductance is maximal when S is low (light condition).
All mGluR6 synapse parameters (g_TRPM1, E_TRPM1, alpha_mGluR6, tau_mGluR6) are
included in the params named tuple.
"""
function on_bipolar_model!(du, u, p, t)
    params, glu_received = p
    V, n, h, c, S, G = u

    # -------- mGluR6 cascade -> TRPM1 gating --------
    # High glutamate (dark) => G≈1 => S→1 => TRPM1 closed
    # Low glutamate (light) => G≈0 => S→0 => TRPM1 open
    S_INF = S_inf(max(glu_received, 0.0), params.K_Glu, params.n_Glu)
    dS = (params.a_S * S_INF - S) / params.tau_S

    # -------- gating dynamics --------
    n_inf = gate_inf(V, params.Vn_half, params.kn_slope)
    dn = (n_inf - n) / params.tau_n

    h_inf = gate_inf(V, params.Vh_half, params.kh_slope) # kh_slope < 0 for Ih
    dh = (h_inf - h) / params.tau_h

    m_inf = gate_inf(V, params.Vm_half, params.km_slope) # CaL activation (instant-ish)
    # if you want dynamic m, swap to a state variable; here we keep it simple

    # -------- currents (I = g * gate * (V - E)) --------
    I_L     = params.g_L * (V - params.E_L)
    I_TRPM1 = params.g_TRPM1 * S * (V - params.E_TRPM1)
    I_Kv    = params.g_Kv * n * (V - params.E_K)
    I_h     = params.g_h * h * (V - params.E_h)
    I_CaL   = params.g_CaL * m_inf * (V - params.E_Ca)

    #-------- Ca pool --------
    # driven by inward Ca current only (when I_CaL is negative)
    Ca_in = max(-I_CaL, 0.0)
    dc = (-c / params.tau_c) + params.k_c * Ca_in

    # KCa activation from Ca pool
    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    I_KCa = params.g_KCa * a_c * (V - params.E_K)

    # -------- voltage derivative --------
    dV = (-I_L - I_TRPM1 - I_Kv - I_h - I_CaL - I_KCa + params.I_app) / params.C_m
    
    # -------- glutamate release --------
    dG = (params.a_Release * R_inf(c, params.K_Release, params.n_Release) - G) / params.tau_Release # dG/dt = (a_Release * R_inf(Ca, K_Release, n_Release) - G) / tau_Release
    du .= (dV, dn, dh, dc, dS, dG)
    return nothing
end
