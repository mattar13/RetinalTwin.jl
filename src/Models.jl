function photoreceptor_model!(du, u, p, t; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
    dV = view(du, 1)
    dRh = view(du, 2)
    dRhi = view(du, 3)
    dTr = view(du, 4)
    dPDE = view(du, 5)
    dCa = view(du, 6)
    dCaB = view(du, 7)
    dcGMP = view(du, 8)

    V = view(u, 1)
    Rh = view(u, 2)
    Rhi = view(u, 3)
    Tr = view(u, 4)
    PDE = view(u, 5)
    Ca = view(u, 6)
    CaB = view(u, 7)
    cGMP = view(u, 8)

    #Extract the parameters
    (   c_m,
        αC, kR_f, kR_r1, kR_r2,
        kT_f, kT_r, Tr_total, 
        kPDE_f, kPDE_r, PDE_total, 
        b, Jmax, kCa_f1, kCa_f2, kCa_r, C0, CaB_total, 
        A_max, kCNG_f, kCNG_r1, kCNG_r2
    ) = p

    @. dV = -(ILeak(V) + Iphoto(V, cGMP, Jmax))/c_m
    @. dRh = αC*Stim(t, stim_start, stim_end, photon_flux) - kR_f*Rh + kR_r1*Rhi
    @. dRhi = kR_f*Rh - (kR_r1 + kR_r2)*Rhi
    @. dTr = kT_f*Rh*(Tr_total - Tr) - kT_r*Tr - kPDE_f*Tr*(PDE_total - PDE)
    @. dPDE = kPDE_f*Tr*(PDE_total - PDE) - kPDE_r*PDE

    @. dCa = b*J(cGMP, Jmax) - kCa_f1*(Ca - C0) - kCa_f2*(CaB_total - CaB)*Ca + kCa_r*CaB
    #@. dCa =  

    @. dCaB = kCa_f2*(CaB_total - CaB)*Ca - kCa_r*CaB

    @. dcGMP = A_max/(1.0 + (Ca/kCNG_f)^4)-cGMP*(kCNG_r1 + kCNG_r2*PDE)
    nothing
end



function phototransduction_ode!(du, u, p, t; stim_start = 0.0, stim_end = 1.0, photon_flux = 400.0)
    #Extract the parameters
    dR = view(du, 1)
    dT = view(du, 2)
    dP = view(du, 3)
    dG = view(du, 4)
    dJ = view(du, 5)
    dH = view(du, 6)

    dA = view(du, 7)

    R = view(u, 1)
    T = view(u, 2)
    P = view(u, 3)
    G = view(u, 4)
    J = view(u, 5)
    H = view(u, 6)

    A = view(u, 7)

    #CONSTANTS
    RMAX = 1.0
    TMAX = 1.0
    PMAX = 1.0
    GMAX = 1.0

    #Open parameters
    (kP1, kP2, kP3, kP4, kP5, kP6,
    l1, h1, JMAX, HMAX,
    τR, τT, τP, τG, τJ, τH) = p

    @. dR = kP1*Stim(t, stim_start, stim_end, photon_flux)*(1-R/RMAX) - R/τR
    @. dT = kP2*R*(1-T/TMAX)- T/τT
    @. dP = kP3*T*(1-P/PMAX) - P/τP
    @. dG = -kP4*P*(1-G/-GMAX) - G/τG # Non-linear degradation
    @. dJ = ( kP5*G*(1-J/-JMAX) - J)/τJ
    @. dH = (kP6*H_inf(J, l1, h1)^2*(1-H/ HMAX) - H)/τH #to add this or not *H_inf(J, l1, h1)

    @. dA = (J+H) - A
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

    @. dB = (-k1*B_inf(A, l2, h2) - B)/τB #This is transfer from PC to BPC #*(-A^3) This nonlinear term may be taking away
    @. dM = (k2*A - k3*B - M)/τM
    @. dC = (-k4*A - C)/τC 
    
    # Oscillator equations with delayed forcing applied to the O1 system:
    @. dO1 = O2
    @. dO2 = (-k5*O1 - k6*O2) + sigm(B, l3, h3)
    @. dO3 = (-k7*B*abs(O2) - O3)/τO
    @. dO = (O3 * O2) - O

    @. dERG = (A + B + M + C + O) - ERG
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

function simulate_model_photo(data, params; kwargs...)
    prob = make_model_photo(data, params; kwargs...)
    sol = solve(prob, Tsit5(), saveat=data.t, tstops=[stim_start, stim_end])
	ERG_t = map(t -> sol(t)[7], data.t)
    return sol, ERG_t
end