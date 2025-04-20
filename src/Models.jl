function phototransduction_ode!(du, u, p, t; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
    #Extract the parameters
    dV = view(du, 1)
    
    dR = view(du, 2)
    dT = view(du, 3)
    dP = view(du, 4)
    dG = view(du, 5)

    dHC1 = view(du, 6)
    dHC2 = view(du, 7)
    dHO1 = view(du, 8)
    dHO2 = view(du, 9)
    dHO3 = view(du, 10)

    dmKV = view(du, 11)
    dhKV = view(du, 12)
    dmCa = view(du, 13)
    dmKCa = view(du, 14)

    d_Ca_s = view(du, 15)
    d_Ca_f = view(du, 16)
    d_CaB_ls = view(du, 17)
    d_CaB_hs = view(du, 18)
    d_CaB_lf = view(du, 19)
    d_CaB_hf = view(du, 20)

    V = view(u, 1)
    
    R = view(u, 2)
    T = view(u, 3)
    P = view(u, 4)
    G = view(u, 5)

    H = view(u, 6:10)
    C1 = view(u, 6)
    C2 = view(u, 7)
    O1 = view(u, 8)
    O2 = view(u, 9)
    O3 = view(u, 10)

    mKV = view(u, 11)
    hKV = view(u, 12)
    mCa = view(u, 13)
    mKCa = view(u, 14)

    _Ca_s = view(u, 15)
    _Ca_f = view(u, 16)
    _CaB_ls = view(u, 17)
    _CaB_hs = view(u, 18)
    _CaB_lf = view(u, 19)
    _CaB_hf = view(u, 20)

    #A = view(u, 8)

    #Open parameters
    (aC, kR1, kF2, kR2, kF3, kR3, kHYDRO, kREC, G0, iDARK, kg, 
    C_m, gLEAK, eLEAK, gH, eH, gKV, eK, gCa, eCa, _Ca_0, gKCa, gCl, eCl, 
    F, DCa, S1, DELTA, V1, V2, Lb1, Bl, Lb2, Hb1, Bh, Hb2,
    J_ex, Cae, K_ex, J_ex2, K_ex2,
    ) = p

    #CONSTANTS
    G0 = 4.0
    kg = 20
    iDARK = 5040.0
    
    #Stimulus
    Φ=Stim(t, stim_start, stim_end, photon_flux)
    
    #Currents
    iLEAK = iH = iKV = iCa = iKCa = iCl = iEX = iEX2 = 0.0 #Initialize the currents to zero
    iPHOTO = @. -iDARK * J∞(G, kg)* (1.0 - exp((V - 8.5) / 17.0))
    iLEAK = @. gLEAK*(V - -eLEAK) #Leak
    iH =    @. gH*(O1 + O2 + O3)*(V - -eH) #Ih Current
    iKV =   @. gKV*mKV^3+hKV*(V - -eK)
    iCa =   @. gCa*mCa^4*hCa(V)*(V - -eCa*log(_Ca_s/_Ca_0)) #Ca current #We should add the log 
    iKCa =  @. gKCa * mKCa^2 * mKCas(_Ca_s) * (V - -eK) #KCa current
    iCl =   @. gCl * mCl(_Ca_s) * (V + eCl) #Cl current
    iEX =   @. J_ex * exp(-(V + 14) / 70) * (_Ca_s - Cae) / ((_Ca_s - Cae) + K_ex)
    iEX2 =  @. J_ex2 * (_Ca_s - Cae) / ((_Ca_s - Cae) + K_ex2)
    
    #Voltage equation
    @. dV = -(iPHOTO + iLEAK + iH + iCa + iCl + iKCa + iKV + iEX + iEX2)/C_m

    #phototransduction equations
    @. dR = aC*Φ - kR1*R
    @. dT = kF2*R*(1-T) - kR2*T
    @. dP = kF3*T*(1-P) - kR3*P
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
    Ca_flux_elec = iCa / (2F*V1) * 1e-6      # µM s⁻¹  (electrogenic)
    Ca_flux_pump = (iEX + iEX2) / (2F*V1) * 1e-6   # export µM s⁻¹
    
    @. d_Ca_s =(-(Ca_flux_elec + Ca_flux_pump) - DCa * (S1 / (DELTA * V1)) * (_Ca_s - _Ca_f) - Lb1 * _Ca_s * (Bl - _CaB_ls) + Lb2 * _CaB_ls - Hb1 * _Ca_s * (Bh - _CaB_hs) + Hb2 * _CaB_hs)
    @. d_Ca_f =(  DCa * (S1 / (DELTA * V2)) * (_Ca_s - _Ca_f) - Lb1 * _Ca_f * (Bl - _CaB_lf) + Lb2 * _CaB_lf - Hb1 * _Ca_f * (Bh - _CaB_hf) + Hb2 * _CaB_hf)
    @. d_CaB_ls = Lb1 * _Ca_s * (Bl - _CaB_ls) - Lb2 * _CaB_ls
    @. d_CaB_hs = Hb1 * _Ca_s * (Bh - _CaB_hs) - Hb2 * _CaB_hs
    @. d_CaB_lf = Lb1 * _Ca_f * (Bl - _CaB_lf) - Lb2 * _CaB_lf
    @. d_CaB_hf = Hb1 * _Ca_f * (Bh - _CaB_hf) - Hb2 * _CaB_hf
    
    #R_m = 10
    #@. dA = -((J+V0)/R_m + H + gREST*(A-0.0))/C_m 
    return nothing
end

function erg_ode!(du, u, p, t; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
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

function make_model(data, params; ex_vivo = true, stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
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

function simulate_model(data, params; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
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