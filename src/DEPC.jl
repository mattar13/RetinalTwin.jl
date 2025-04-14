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

    GMAX = 1.0

    #Open parameters
    (kR, kP2, kP3, kP4, kP5, kP6,
    l1, h1, JMAX, HMAX,
    τR, τT, τP, τG, τJ, τH) = p

    Φ=Stim(t, stim_start, stim_end, photon_flux)
    @. dR = kP1*Φ#=*(1-R)=# - R/τR
    @. dT = kP2*R*(1-T)- T/τT
    @. dP = kP3*T*(1-P) - P/τP
    @. dG = -kP4*P*(1-G/-GMAX) - G/τG # Non-linear degradation

    @. dJ = ( kP5*G*(1-J/-JMAX) - J)/τJ
    @. dH = (kP6*H_inf(J, l1, h1)^2*(1-H/ HMAX) - H)/τH #to add this or not *H_inf(J, l1, h1)

    @. dA = (J+H) - A
    return nothing
end