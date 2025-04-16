function phototransduction_ode!(du, u, p, t; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
    #Extract the parameters
    dR = view(du, 1)
    dT = view(du, 2)
    dP = view(du, 3)
    dG = view(du, 4)
    dIPHOTO = view(du, 5)
    dH = view(du, 6)

    dA = view(du, 7)

    R = view(u, 1)
    T = view(u, 2)
    P = view(u, 3)
    G = view(u, 4)
    IPHOTO = view(u, 5)
    H = view(u, 6)

    A = view(u, 7)

    #Open parameters
    (aC, kR1, kF2, kR2, kF3, kR3,
    kHYDRO, kREC, G0, 

    iDARK, kg, C_m,
    V0, gREST, l1, h1,
    gH, τH) = p

    #CONSTANTS
    G0 = 4.0
    kg = 20.0
    iDARK = 5040.0

    Φ=Stim(t, stim_start, stim_end, photon_flux)
    @. dR = aC*Φ - kR1*R
    @. dT = kF2*R*(1-T) - kR2*T
    @. dP = kF3*T*(1-P) - kR3*P
    @. dG = -kHYDRO*P*G + kREC*(G0 - G) # Non-linear degradation

    @. dIPHOTO = -iDARK * J∞(G, kg) - J#( kP5*G*(1-J/-JMAX) - J)/τJ
    @. dH = (gH*H_inf(A, l1, h1)*(A-0.25) - H)/τH #0.0#(kP6*H_inf(J, l1, h1)^2*(1-H/ HMAX) - H)/τH #to add this or not *H_inf(J, l1, h1)
    R_m = 10
    E_REST = 0.0
    
    iALL = 
    @. dV = -((J+V0)/R_m + H + gREST*(A-E_REST))/C_m
    
    
    
    @. dA = -((J+V0)/R_m + H + gREST*(A-E_REST))/C_m 



    return nothing
end

function erg_ode!(du, u, p, t; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
    dB = view(du, 8)
    dM = view(du, 9)
    dC = view(du, 10)

    dO1 = view(du, 11)
    dO2 = view(du, 12)
    dO3 = view(du, 13)
    dO = view(du, 14)
    
    dERG = view(du, 15)
    
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