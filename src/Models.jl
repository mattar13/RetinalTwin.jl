function phototransduction_ode!(du, u, p, t; stim_start = 0.0, stim_end = 0.0, photon_flux = 0.0, v_hold = false)
    #Extract the parameters
    
    dR = view(du, 1)
    dT = view(du, 2)
    dP = view(du, 3)
    dG = view(du, 4)
    
    dHC1 = view(du, 5)
    dHC2 = view(du, 6)
    dHO1 = view(du, 7)
    dHO2 = view(du, 8)
    dHO3 = view(du, 9)
    
    dmKV = view(du, 10)
    dhKV = view(du, 11)
    dmCa = view(du, 12)
    dmKCa = view(du, 13)
    
    d_Ca_s = view(du, 14)
    d_Ca_f = view(du, 15)
    d_CaB_ls = view(du, 16)
    d_CaB_hs = view(du, 17)
    d_CaB_lf = view(du, 18)
    d_CaB_hf = view(du, 19)
    
    dV = view(du, 20)
    
    R = view(u, 1)
    T = view(u, 2)
    P = view(u, 3)
    G = view(u, 4)
    
    H = view(u, 5:9)
    O1 = view(u, 7)
    O2 = view(u, 8)
    O3 = view(u, 9)
    
    mKV = view(u, 10)
    hKV = view(u, 11)
    mCa = view(u, 12)
    mKCa = view(u, 13)
    
    _Ca_s = view(u, 14)
    _Ca_f = view(u, 15)
    _CaB_ls = view(u, 16)
    _CaB_hs = view(u, 17)
    _CaB_lf = view(u, 18)
    _CaB_hf = view(u, 19)
    
    V = view(u, 20)

    #Open parameters
    (aC, kR1, kF1, kR2, kR3, kHYDRO, kREC, G0, iDARK, kg, 
    C_m, gLEAK, eLEAK, gH, eH, gKV, eK, gCa, eCa, _Ca_0, gKCa, gCl, eCl, 
    F, DCa, S1, DELTA, V1, V2, Lb1, Bl, Lb2, Hb1, Bh, Hb2,
    J_ex, Cae, K_ex, J_ex2, K_ex2,
    ) = p

    #CONSTANTS
    G0 = 2.0
    kg = 10.0
    iDARK = 5040.0
    
    #Stimulus
    if v_hold
        V_HOLD = Stim(t, stim_start, stim_end, photon_flux; hold = -36.186)
        Φ = 0.0
    else
        Φ = Stim(t, stim_start, stim_end, photon_flux)
    end

    #Reversal potentials (- sign only once) -------------
    E_LEAK   = -eLEAK
    E_H   = -eH
    E_K   = -eK
    E_Cl  = -eCl
    E_Ca =  @. eCa * log(_Ca_0 / max(_Ca_s, 1e-5)) 

    #Currents
    iLEAK = iH = iKV = iCa = iKCa = iCl = iEX = iEX2 = 0.0 #Initialize the currents to zero
    iPHOTO = @. -iDARK * J∞(G, 10.0) * (1.0 - exp((V - 8.5) / 17.0))
    iLEAK = @. gLEAK*(V - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V - E_K)
    iCa =   @. gCa*mCa^4*hCa(V)*(V - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (V - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)
    

    #phototransduction equations
    R_tot = 3.0 #mM
    T_tot = 0.3 #mM
    P_tot = 0.021 #mM
    λ = 0.67 #This is constant quantum efficiency
    @. dR = aC*λ*Φ*(R_tot-R) - kR1*R*(T_tot-T) - kF1*R
    @. dT = kR1*R*(T_tot-T) - kR2*T*(P_tot-P)
    @. dP = kR2*T*(P_tot-P) - kR3*P
    @. dG = -kHYDRO*P*G + kREC*(G0 - G) # Non-linear degradation
    
    #Hyperpolarization-activated current (Ih) equations
    rH = hT.(V) * H
    @. dHC1 = rH[1]
    @. dHC2 = rH[2]
    @. dHO1 = rH[3]
    @. dHO2 = rH[4]
    @. dHO3 = rH[5]

    #Channel gating equations
    @. dmKV = αmKV(V) * (1 - mKV) - βmKV(V) * mKV
    @. dhKV = αhKV(V) * (1 - hKV) - βhKV(V) * hKV
    @. dmCa = αmCa(V) * (1 - mCa) - βmCa(V) * mCa
    @. dmKCa = αmKCa(V) * (1 - mKCa) - βmKCa(V) * mKCa

    #Calcium dynamics
    Ca_flux = -(iCa + iEX + iEX2) / (2F*V1) * 1e-6   # export µM s⁻¹
    @. d_Ca_s =(Ca_flux - DCa * (S1 / (DELTA * V1)) * (_Ca_s - _Ca_f) - Lb1 * _Ca_s * (Bl - _CaB_ls) + Lb2 * _CaB_ls - Hb1 * _Ca_s * (Bh - _CaB_hs) + Hb2 * _CaB_hs)
    @. d_Ca_f =(  DCa * (S1 / (DELTA * V2)) * (_Ca_s - _Ca_f) - Lb1 * _Ca_f * (Bl - _CaB_lf) + Lb2 * _CaB_lf - Hb1 * _Ca_f * (Bh - _CaB_hf) + Hb2 * _CaB_hf)
    @. d_CaB_ls = Lb1 * _Ca_s * (Bl - _CaB_ls) - Lb2 * _CaB_ls
    @. d_CaB_hs = Hb1 * _Ca_s * (Bh - _CaB_hs) - Hb2 * _CaB_hs
    @. d_CaB_lf = Lb1 * _Ca_f * (Bl - _CaB_lf) - Lb2 * _CaB_lf
    @. d_CaB_hf = Hb1 * _Ca_f * (Bh - _CaB_hf) - Hb2 * _CaB_hf
    #println(t)
    #R_m = 10
    #Voltage equation
    if v_hold
        @. dV = (V_HOLD - V)/C_m
    else
        @. dV = -(iPHOTO + iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/C_m
    end
    #@. dA = -((J+V0)/R_m + H + gREST*(A-0.0))/C_m 
    return nothing
end


function phototransduction_compartments!(du, u, p, t; stim_start = 0.0, stim_end = 0.0, photon_flux = 0.0, v_hold = false)
    #Extract the parameters
    
    phototransduction_ode!(du, u, p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux, v_hold = v_hold)
    
    G = view(u, 4)
    O1 = view(u, 7)
    O2 = view(u, 8)
    O3 = view(u, 9)
    mKV = view(u, 10)
    hKV = view(u, 11)
    mCa = view(u, 12)
    mKCa = view(u, 13)
    _Ca_s = view(u, 14)
    
    V_os    = view(u, 20)
    V_osis  = view(u, 21)
    V_is    = view(u, 22)
    V_iscb  = view(u, 23)
    V_ax    = view(u, 24)
    V_st    = view(u, 25)
    
    dV_os    = view(du, 20)
    dV_osis  = view(du, 21)
    dV_is    = view(du, 22)
    dV_iscb  = view(du, 23)
    dV_ax    = view(du, 24)
    dV_st    = view(du, 25)

    du_os = view(du, 26)
    du_osis = view(du, 27)
    du_is = view(du, 28)
    du_iscb = view(du, 29)
    du_ax = view(du, 30)
    du_st = view(du, 31)

    u_os = view(u, 26)
    u_osis = view(u, 27)
    u_is = view(u, 28)
    u_iscb = view(u, 29)
    u_ax = view(u, 30)
    u_st = view(u, 31)

    #Open parameters
    (aC, kR1, kF1, kR2, kR3, kHYDRO, kREC, G0, iDARK, kg, 
    C_m, gLEAK, eLEAK, gH, eH, gKV, eK, gCa, eCa, _Ca_0, gKCa, gCl, eCl, 
    F, DCa, S1, DELTA, V1, V2, Lb1, Bl, Lb2, Hb1, Bh, Hb2,
    J_ex, Cae, K_ex, J_ex2, K_ex2,
    ) = p

    #CONSTANTS
    iDARK = 5040.0
    Cm_os = 46.02
    Cm_osis = 0.236
    Cm_is = 43.21
    Cm_iscb = 6.43
    Cm_ax = 9.09
    Cm_st = 4.19

    #Reversal potentials (- sign only once) -------------
    E_LEAK   = -eLEAK
    E_H   = -eH
    E_K   = -eK
    E_Cl  = -eCl
    E_Ca =  @. eCa * log(_Ca_0 / max(_Ca_s, 1e-5)) 

    #Currents
    iLEAK = @. gLEAK*(V - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V - E_K)
    iCa =   @. gCa*mCa^4*hCa(V)*(V - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (V - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)

    iOSIS = iLEAK
    iIS = iISCB = iCB = iAX = iST = iCa + iCl + iEX + iEX2 + iKCa + iH + iKV

    #Voltage equation
    iPHOTO = @. -iDARK * J∞(G, 10.0) * (1.0 - exp((V_os - 8.5) / 17.0))
    @. dV_os = -(iPHOTO)/Cm_os

    iLEAK = @. gLEAK*(V_osis - E_LEAK) #Leak
    @. dV_osis = -(iLEAK)/Cm_osis

    iLEAK = @. gLEAK*(V_is - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V_is - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V_is - E_K)
    iCa =   @. gCa*mCa^4*hCa(V_is)*(V_is - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (V_is - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V_is - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V_is + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)
    @. dV_is = -(iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/Cm_is

    iLEAK = @. gLEAK*(V_iscb - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V_iscb - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V_iscb - E_K)
    iCa =   @. gCa*mCa^4*hCa(V_iscb)*(V_iscb - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (V_is - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V_iscb - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V_iscb + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)
    @. dV_iscb = -(iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/Cm_iscb

    iLEAK = @. gLEAK*(V_cb - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V_cb - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V_cb - E_K)
    iCa =   @. gCa*mCa^4*hCa(V_cb)*(V_cb - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (cb - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V_cb - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V_cb + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)
    @. dV_cb = -(iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/Cm_cb
    
    iLEAK = @. gLEAK*(V_ax - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V_ax - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V_ax - E_K)
    iCa =   @. gCa*mCa^4*hCa(V_ax)*(V_ax - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (cb - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V_ax - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V_ax + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)
    @. dV_ax = -(iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/Cm_ax

    iLEAK = @. gLEAK*(V_st - E_LEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V_st - E_H) #Ih Current
    iKV =   @. gKV*mKV^3*hKV*(V_st - E_K)
    iCa =   @. gCa*mCa^4*hCa(V_st)*(V_st - E_Ca) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (cb - E_K) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V_st - E_Cl) #Cl current
    iEX =   @. J_ex * C∞(_Ca_s, Cae, K_ex) * exp(-(V_st + 14) / 70)
    iEX2 =  @. J_ex2 * C∞(_Ca_s, Cae, K_ex2)
    @. dV_st = -(iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/Cm_st
    return nothing
end

    
function erg_ode!(du, u, p, t; stim_start = 0.0, stim_end = 0.0, photon_flux = 0.0)
    dB = view(du, 9)
    dM = view(du, 10)
    dC = view(du, 11)

    dO1 = view(du, 12)
    dO2 = view(du, 13)
    dO3 = view(du, 14)
    dO = view(du, 15)
    
    dERG = view(du, 16)
    
    A = view(u, 7)
    B = view(u, 8)
    M = view(u, 9)
    C = view(u, 10)

    O1 = view(u, 11)
    O2 = view(u, 12)
    O3 = view(u, 13)
    O = view(u, 14)
    
    ERG = view(u, 15)
    
    #Open parameters
    (photo_p..., #Splat the first several params away
    l2, h2,
    k1, k2, k3, k4, k5, k6, k7, 
    l3, h3, 
    τB, τM, τC, τO) = p
    
    #Run phototransduction first
    phototransduction_ode!(du, u, photo_p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux)

    @. dB = (k1*(abs(A)^4) - B)/τB #This is transfer from PC to BPC #*(-A^3) This nonlinear term may be taking away
    @. dM = (k2*A - k3*B - M)/τM
    @. dC = (-k4*A - C)/τC 
    
    # Oscillator equations with delayed forcing applied to the O1 system:
    @. dO1 = O2
    @. dO2 = (-k5*O1 - k6*O2) + sigm(B, l3, h3)
    @. dO3 = (-k7*B*abs(O2) - O3)/τO
    @. dO = (O3 * O2) - O

    @. dERG = (A + B + M) - ERG # + C + O
    nothing
end

function make_model(data, params; ex_vivo = true, stim_start = 0.0, stim_end = 0.0, photon_flux = 0.0)
    tspan = (0.0, data.t[end])
    u0 = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0
        ]

    u0[4] = 4.0
    u0[5] = -40.0
    if ex_vivo
        params[22] = 0.0 #Silence k4 
        params[25] = 0.0 #Silence k7
    end

    prob = ODEProblem(
        (du, u, p, t) -> erg_ode!(du, u, p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux), 
        u0, tspan, params
    )
    return prob
end

function simulate_model(data, params; stim_start = 0.0, stim_end = 0.0, photon_flux = 0.0)
    prob = make_model(data, params; stim_start = stim_start, stim_end = stim_end,photon_flux = photon_flux)
    sol = solve(prob, Tsit5(), saveat=data.t, tstops=[stim_start, stim_end])
	ERG_t = map(t -> sol(t)[15], data.t)
    return sol, ERG_t
end

#DO this for the phototransduction model only
function make_model_photo(data, params; ex_vivo = true, kwargs...)
    tspan = (0.0, data.t[end])
    u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    prob = ODEProblem(
        (du, u, p, t) -> phototransduction_ode!(du, u, p, t; kwargs...), 
        u0, tspan, params
    )
    return prob
end

function simulate_model_photo(data, params; stim_start = 0.0, stim_end = 1.0, kwargs...)
    prob = make_model_photo(data, params; stim_start = stim_start, stim_end = stim_end, kwargs...)
    sol = solve(prob, Tsit5(), saveat=data.t, tstops=[stim_start, stim_end])
	ERG_t = map(t -> sol(t)[7], data.t)
    return sol, ERG_t
end