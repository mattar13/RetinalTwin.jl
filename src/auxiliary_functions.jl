# ============================================================
# auxiliary_functions.jl - Shared helper functions across cell models
# ============================================================

const RT_F = 26.7  # mV at 37C (RT/F for ln)

# Generic nonlinearities
@inline σ(x) = 1.0 / (1.0 + exp(-x))

@inline function gate_inf(V, Vhalf, k)
    return 1.0 / (1.0 + exp(-(V - Vhalf) / k))
end

@inline function hill(x, K, n)
    xn = x^n
    return xn / (K^n + xn + eps())
end

"""
    spatial_synaptic(release_sites, weights, params, relation, K_key, n_key)

Compute weighted-average channel activation across presynaptic release sites.
`relation` is one of `:hill` or `:inv_hill`; `K_key`/`n_key` select fields in `params`.
"""
function spatial_synaptic(
    release_sites::AbstractVector,
    weights::AbstractVector,
    params,
    relation::Symbol,
    K_key::Symbol,
    n_key::Symbol
)
    length(release_sites) == length(weights) || error(
        "spatial_synaptic: release_sites and weights must have same length"
    )
    isempty(release_sites) && return 0.0

    wsum = sum(weights)
    wsum <= 0 && return 0.0

    K = getproperty(params, K_key)
    n = getproperty(params, n_key)

    acc = 0.0
    @inbounds for i in eachindex(release_sites, weights)
        x = max(release_sites[i], 0.0)
        a = if relation == :hill
            hill(x, K, n)
        elseif relation == :inv_hill
            1.0 / (1.0 + (x / K)^n)
        else
            error("spatial_synaptic: unsupported relation $relation")
        end
        acc += weights[i] * a
    end
    return acc / (wsum + eps())
end

# Bipolar/amacrine release and receptor helpers
@inline S_inf(glu_received, K_Glu, n_Glu) = 1.0 / (1.0 + (glu_received / K_Glu)^n_Glu)
@inline R_inf(Ca, K_Release, n_Release) = (Ca^n_Release) / (K_Release^n_Release + Ca^n_Release + eps())
@inline A_inf(glu, K, n) = hill(max(glu, 0.0), K, n)
@inline D_inf(glu, K, n) = 1.0 / (1.0 + (max(glu, 0.0) / K)^n)

# Morris-Lecar helpers
@inline m_inf_ml(V, V1, V2) = 0.5 * (1.0 + tanh((V - V1) / V2))
@inline w_inf_ml(V, V3, V4) = 0.5 * (1.0 + tanh((V - V3) / V4))
@inline tau_w_ml(V, V3, V4) = 1.0 / cosh((V - V3) / (2.0 * V4))

function hc_feedback(V_hc::Real; g_FB::Real=1.0, V_FB_half::Real=-50.0, V_FB_slope::Real=5.0)
    return g_FB / (1.0 + exp(-(V_hc - V_FB_half) / V_FB_slope))
end

# Ganglion HH gating helpers
@inline function alpha_beta_m(V)
    ϕ = V + 40.0
    α = (abs(ϕ) < 1e-6) ? 1.0 : 0.1 * ϕ / (1 - exp(-0.1 * ϕ))
    β = 4.0 * exp(-(V + 65.0) / 18.0)
    return α, β
end

@inline function alpha_beta_h(V)
    α = 0.07 * exp(-(V + 65.0) / 20.0)
    β = 1.0 / (1 + exp(-0.1 * (V + 35.0)))
    return α, β
end

@inline function alpha_beta_n(V)
    ϕ = V + 55.0
    α = (abs(ϕ) < 1e-6) ? 0.1 : 0.01 * ϕ / (1 - exp(-0.1 * ϕ))
    β = 0.125 * exp(-(V + 65.0) / 80.0)
    return α, β
end

# K+ equilibrium / Kir helpers
@inline nernst_K(K_o::Real, K_i::Real) = RT_F * log(max(K_o, 1e-6) / max(K_i, 1e-6))

@inline function kir_rect(V::Real, E_K::Real; Vshift::Real=15.0, k::Real=6.0)
    return 1.0 / (1.0 + exp((V - (E_K + Vshift)) / k))
end

@inline function mBK_inf(V::Real, Ca::Real; Vhalf0::Real=-10.0, k::Real=8.0, s::Real=20.0, Caref::Real=0.1)
    Ca_safe = max(Ca, 1e-9)
    Caref_safe = max(Caref, 1e-9)
    Vhalf = Vhalf0 - s * log(Ca_safe / Caref_safe)
    return 1.0 / (1.0 + exp(-(V - Vhalf) / k))
end

# Photoreceptor helpers
@inline Stim(t, t_on, t_off, Phi; hold=0) = (t_on <= t <= t_off ? Phi : hold)
@inline J∞(g, kg) = g^3 / (g^3 + kg^3)
@inline C∞(C, Cae, K) = C > Cae ? (C - Cae) / (C - Cae + K) : 0.0

@inline αmKv(v) = 5 * (100 - v) / (exp((100 - v) / 42) - 1)
@inline βmKv(v) = 9 * exp(-(v - 20) / 40)
@inline αhKv(v) = 0.15 * exp(-v / 22)
@inline βhKv(v) = 0.4125 / (exp((10 - v) / 7) + 1)

@inline αmCa(v) = 3 * (80 - v) / (exp((80 - v) / 25) - 1)
@inline βmCa(v) = 10 / (1 + exp((v + 38) / 7))
@inline hCa(v) = exp((40 - v) / 18) / (1 + exp((40 - v) / 18))

@inline αmKCa(v) = 15 * (80 - v) / (exp((80 - v) / 40) - 1)
@inline βmKCa(v) = 20 * exp(-v / 35)
@inline mKCas(C) = C / (C + 0.3)

@inline mCl(C) = 1 / (1 + exp((0.37 - C) / 0.09))

@inline αh(v) = 8 / (exp((v + 78) / 14) + 1)
@inline βh(v) = 18 / (exp(-(v + 8) / 19) + 1)

@inline R_glu_inf(V, params) = params.alpha_Glu / (1.0 + exp(-(V - params.V_Glu_half) / params.V_Glu_slope))

function hT(v)
    α = αh(v)
    β = βh(v)
    return [
        -4α      β       0       0       0
         4α  -(3α + β)   2β      0       0
         0      3α   -(2α + 2β)  3β      0
         0       0      2α   -(α + 3β)   4β
         0       0       0       α     -4β
    ]
end
