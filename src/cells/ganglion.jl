# ============================================================
# ganglion.jl - Ganglion cell dynamics
# ============================================================

# ── 2. Initial Conditions ───────────────────────────────────

"""
    ganglion_dark_state(params)

Return dark-adapted initial conditions for a ganglion cell.

# Arguments
- `params`: named tuple from `default_gc_params()`

# Returns
- 2-element state vector [V, w]
"""
function ganglion_dark_state(params)
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

# Use shared ML functions from horizontal.jl
@inline σ(x) = 1.0 / (1.0 + exp(-x))

@inline function alpha_beta_m(V)
    # V in mV; classic HH shifted to mV-scale
    ϕ = V + 40.0
    α = (abs(ϕ) < 1e-6) ? 1.0 : 0.1*ϕ/(1 - exp(-0.1*ϕ))
    β = 4.0*exp(-(V + 65.0)/18.0)
    return α, β
end

@inline function alpha_beta_h(V)
    α = 0.07*exp(-(V + 65.0)/20.0)
    β = 1.0/(1 + exp(-0.1*(V + 35.0)))
    return α, β
end

@inline function alpha_beta_n(V)
    ϕ = V + 55.0
    α = (abs(ϕ) < 1e-6) ? 0.1 : 0.01*ϕ/(1 - exp(-0.1*ϕ))
    β = 0.125*exp(-(V + 65.0)/80.0)
    return α, β
end

#Using the hill function that was already defined in on_bipolar.jl
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

# ── 4. Mathematical Model ───────────────────────────────────

"""
    ganglion_model!(du, u, p, t)

Morris-Lecar ganglion cell model.

# Arguments
- `du`: derivative vector (2 elements)
- `u`: state vector (2 elements)
- `p`: tuple `(params, I_exc, I_inh)` where:
  - `params`: named tuple from `default_gc_params()`
  - `I_exc`: excitatory synaptic current from bipolars (pA)
  - `I_inh`: inhibitory synaptic current from amacrines (pA)
- `t`: time (ms)

# State vector
`u = [V, w]`

# Notes
Output neuron for action potential generation. Receives excitatory
input from ON/OFF bipolars and inhibitory input from amacrines.
"""
function ganglion_model!(du, u, p, t)
    params, glu_in, gly_in = p
    V, m, h, n, sE, sI = u

    # ----- presynaptic drive -> synaptic open probability targets -----
    # Treat glu_in and gly_in as "release proxies" in [0,1] or arbitrary;
    # map with Hill to transmitter effectiveness:
    preE = hill01(glu_in, params.K_preE, params.n_preE)
    preI = hill01(gly_in, params.K_preI, params.n_preI)

    # First-order synapse gating (conductance-based)
    # ds/dt = (s_inf(pre) - s)/tau
    sE_inf = preE
    sI_inf = preI
    dsE = (sE_inf - sE) / params.tau_E
    dsI = (sI_inf - sI) / params.tau_I

    # ----- intrinsic currents (HH) -----
    I_L  = params.g_L  * (V - params.E_L)
    I_Na = params.g_Na * (m^3) * h * (V - params.E_Na)
    I_K  = params.g_K  * (n^4) * (V - params.E_K)

    # ----- synaptic currents -----
    I_E = params.g_E * sE * (V - params.E_E)   # AMPA-ish, E_E ~ 0 mV
    I_I = params.g_I * sI * (V - params.E_I)   # Gly/GABA, E_I ~ -70 mV

    # ----- membrane equation -----
    dV = (-I_L - I_Na - I_K - I_E - I_I) / params.C_m

    # ----- gating dynamics -----
    αm, βm = alpha_beta_m(V)
    αh, βh = alpha_beta_h(V)
    αn, βn = alpha_beta_n(V)

    dm = αm*(1 - m) - βm*m
    dh = αh*(1 - h) - βh*h
    dn = αn*(1 - n) - βn*n

    du .= (dV, dm, dh, dn, dsE, dsI)
    return nothing
end