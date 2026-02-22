# ============================================================
# current_equations.jl - Centralized membrane current equations
# ============================================================

# Conventions:
# - Function names follow: [cell]_I_[current_name](variables..., params)
# - Variables are passed as individual positional arguments.

# -------------------------
# Photoreceptor currents
# -------------------------
@inline photoreceptor_I_photo(V, G, params) =
    -params.iDARK * J∞(G, 10.0) * (1.0 - exp((V - 8.5) / 17.0))

@inline photoreceptor_I_leak(V, params) = params.gLEAK * (V - (-params.ELEAK))

@inline photoreceptor_I_h(V, HO_sum, params) =
    params.gH * HO_sum * (V - (-params.eH))

@inline photoreceptor_I_kv(V, mKv, hKv, params) =
    params.gKV * mKv^3 * hKv * (V - (-params.eK))

@inline function photoreceptor_I_ca(V, mCa, Ca_s, params)
    E_Ca = params.eCa * log(params._Ca_0 / max(Ca_s, 1e-5))
    return params.gCa * mCa^4 * hCa(V) * (V - E_Ca)
end

@inline photoreceptor_I_kca(V, mKCa, Ca_s, params) =
    params.gKCa * mKCa^2 * mKCas(Ca_s) * (V - (-params.eK))

@inline photoreceptor_I_cl(V, Ca_s, params) =
    params.gCl * mCl(Ca_s) * (V - (-params.eCl))

@inline photoreceptor_I_ex(V, Ca_s, params) =
    params.J_ex * C∞(Ca_s, params.Cae, params.K_ex) * exp(-(V + 14) / 70)

@inline photoreceptor_I_ex2(Ca_s, params) =
    params.J_ex2 * C∞(Ca_s, params.Cae, params.K_ex2)

# -------------------------
# Horizontal cell currents
# -------------------------
@inline horizontal_I_leak(V, params) = params.g_L * (V - params.E_L)

@inline horizontal_I_exc(V, s_inf, params) =
    params.g_exc * s_inf * (V - params.E_exc)

@inline horizontal_I_cal(V, mCa, params) =
    params.g_CaL * mCa * (V - params.E_Ca)

@inline function horizontal_I_kir(V, params)
    r_kir = kir_rect(V, params.E_Kir; Vshift=params.Kir_Vshift, k=params.Kir_k)
    return params.g_Kir * r_kir * (V - params.E_Kir)
end

@inline function horizontal_I_bk(V, c, params)
    m = mBK_inf(V, c; Vhalf0=params.Vhalf0_BK, k=params.k_BK, s=params.s_BK, Caref=params.Caref_BK)
    return params.gBK * m * (V - params.E_K)
end

# -------------------------
# ON bipolar currents
# -------------------------
@inline on_bipolar_I_leak(V, params) = params.g_L * (V - params.E_L)
@inline on_bipolar_I_trpm1(V, S, params) = params.g_TRPM1 * S * (V - params.E_TRPM1)
@inline on_bipolar_I_kv(V, n, params) = params.g_Kv * n * (V - params.E_K)
@inline on_bipolar_I_h(V, h, params) = params.g_h * h * (V - params.E_h)
@inline on_bipolar_I_cal(V, m, params) = params.g_CaL * m * (V - params.E_Ca)

@inline function on_bipolar_I_kca(V, c, params)
    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    return params.g_KCa * a_c * (V - params.E_K)
end

# -------------------------
# OFF bipolar currents
# -------------------------
@inline off_bipolar_I_leak(V, params) = params.g_L * (V - params.E_L)
@inline off_bipolar_I_iglu(V, A, D, params) =
    params.g_iGluR * (A * D) * (V - params.E_iGluR)
@inline off_bipolar_I_kv(V, n, params) = params.g_Kv * n * (V - params.E_K)
@inline off_bipolar_I_h(V, h, params) = params.g_h * h * (V - params.E_h)
@inline off_bipolar_I_cal(V, m, params) = params.g_CaL * m * (V - params.E_Ca)

@inline function off_bipolar_I_kca(V, c, params)
    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    return params.g_KCa * a_c * (V - params.E_K)
end

# -------------------------
# A2 amacrine currents
# -------------------------
@inline a2_amacrine_I_leak(V, params) = params.g_L * (V - params.E_L)
@inline a2_amacrine_I_iglu(V, A, D, params) =
    params.g_iGluR * (A * D) * (V - params.E_iGluR)
@inline a2_amacrine_I_kv(V, n, params) = params.g_Kv * n * (V - params.E_K)
@inline a2_amacrine_I_h(V, h, params) = params.g_h * h * (V - params.E_h)
@inline a2_amacrine_I_cal(V, m, params) = params.g_CaL * m * (V - params.E_Ca)

@inline function a2_amacrine_I_kca(V, c, params)
    a_c = hill(max(c, 0.0), params.K_c, params.n_c)
    return params.g_KCa * a_c * (V - params.E_K)
end

# -------------------------
# GABA amacrine currents
# -------------------------
@inline gaba_amacrine_I_leak(V, params) = params.g_L * (V - params.E_L)
@inline gaba_amacrine_I_ca(V, m, params) = params.g_Ca * m * (V - params.E_Ca)
@inline gaba_amacrine_I_k(V, w, params) = params.g_K * w * (V - params.E_K)

# -------------------------
# DA amacrine currents
# -------------------------
@inline da_amacrine_I_leak(V, params) = params.g_L * (V - params.E_L)
@inline da_amacrine_I_ca(V, m, params) = params.g_Ca * m * (V - params.E_Ca)
@inline da_amacrine_I_k(V, w, params) = params.g_K * w * (V - params.E_K)

# -------------------------
# Ganglion currents
# -------------------------
@inline ganglion_I_leak(V, params) = params.g_L * (V - params.E_L)
@inline ganglion_I_na(V, m, h, params) = params.g_Na * (m^3) * h * (V - params.E_Na)
@inline ganglion_I_k(V, n, params) = params.g_K * (n^4) * (V - params.E_K)
@inline ganglion_I_exc(V, sE, params) = params.g_E * sE * (V - params.E_E)
@inline ganglion_I_inh(V, sI, params) = params.g_I * sI * (V - params.E_I)

# -------------------------
# Muller currents
# -------------------------
@inline function muller_I_kir_end(V, K_o_end, params)
    E_K_end = nernst_K(K_o_end, params.K_i)
    r_end = kir_rect(V, E_K_end; Vshift=params.Kir_Vshift, k=params.Kir_k)
    return params.g_Kir_end * r_end * (V - E_K_end)
end

@inline function muller_I_kir_stalk(V, K_o_stalk, params)
    E_K_stalk = nernst_K(K_o_stalk, params.K_i)
    r_stalk = kir_rect(V, E_K_stalk; Vshift=params.Kir_Vshift, k=params.Kir_k)
    return params.g_Kir_stalk * r_stalk * (V - E_K_stalk)
end

@inline muller_I_leak(V, params) = params.g_L * (V - params.E_L)

@inline function muller_I_eaat(Glu_o, params)
    Glu_o_clamped = max(Glu_o, 0.0)
    J_uptake = params.V_max_EAAT * hill(Glu_o_clamped, params.K_m_EAAT, params.n_EAAT)
    return -params.g_EAAT * J_uptake
end

# -------------------------
# RPE currents
# -------------------------
@inline function rpe_I_k_apical(V, K_sub, params)
    E_K_sub = nernst_K(K_sub, params.K_i)
    return params.g_K_apical * (V - E_K_sub)
end

@inline rpe_I_cl_basal(V, params) = params.g_Cl_baso * (V - params.E_Cl)
@inline rpe_I_leak(V, params) = params.g_L_RPE * (V - params.E_L_RPE)