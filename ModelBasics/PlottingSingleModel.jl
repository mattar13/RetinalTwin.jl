#Model imports
import DigitalTwin: hCa, mCl, mKCas, J∞, C∞

#Define a convienance function
function opts(i; rng = :log10) 
    if rng == :log10
        return (
            color = round(log10(stim_range[i])), 
            colormap = :viridis, 
            colorrange = (log10(stim_range[1]), log10(stim_range[end]))
        )
    else
        return (
            color = round(stim_range[i]), 
            colormap = :viridis, 
            colorrange = (stim_range[1], stim_range[end])
        )
    end
    return nothing  
end
        
rng = :log10
#%% Plotting Figure 1 phototransduction
fig1 = Figure(size = (1200, 600), title = "Phototransduction model")
ax1a = Axis(fig1[1, 1], title = "Rho"); hidespines!(ax1a)
ax1b = Axis(fig1[1, 2], title = "Tr"); hidespines!(ax1b)
ax2a = Axis(fig1[2, 1], title = "PDE"); hidespines!(ax2a)
ax2b = Axis(fig1[2, 2], title = "cGMP"); hidespines!(ax2b)
ax3 = Axis(fig1[1:2, 3], title = "iPHOTO"); hidespines!(ax3)

for (i, sol) in enumerate(data_series)
    V_t = map(t -> sol(t)[20], t_rng) #This is the voltage equation, but we are not using it in this model

    R_t = map(t -> sol(t)[1], t_rng)
    T_t = map(t -> sol(t)[2], t_rng)
    P_t = map(t -> sol(t)[3], t_rng)
    G_t = map(t -> sol(t)[4], t_rng)

    iPHOTO = @. -iDARK * J∞(G_t, kg) * (1.0 - exp((V_t - 8.5) / 17.0))

    vlines!(ax1a, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax1a, t_rng, R_t; opts(i, rng = rng)...)

    vlines!(ax1b, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax1b, t_rng, T_t; opts(i, rng = rng)...)

    vlines!(ax2a, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2a, t_rng, P_t; opts(i, rng = rng)...)

    vlines!(ax2b, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2b, t_rng, G_t; opts(i, rng = rng)...)

    vlines!(ax3, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax3, t_rng, iPHOTO; opts(i, rng = rng)...)
end
fig1

#%% Show all of the currents in the model
fig2 = Figure(size = (1200, 600), title = "Electrical Equivalence Circuit")
ax1 = Axis(fig2[1:2, 1], title = "V_t"); hidespines!(ax1)
ax2a = Axis(fig2[1, 2], title = "iPHOTO"); hidespines!(ax2a)
ax2b = Axis(fig2[1, 3], title = "iLEAK"); hidespines!(ax2b)
ax2c = Axis(fig2[1, 4], title = "iH"); hidespines!(ax2c)
ax2d = Axis(fig2[1, 5], title = "iKV"); hidespines!(ax2d)
ax3a = Axis(fig2[2, 2], title = "iCa"); hidespines!(ax2a)
ax3b = Axis(fig2[2, 3], title = "iKCa"); hidespines!(ax2b)
ax3c = Axis(fig2[2, 4], title = "iCl"); hidespines!(ax2c)

ax4a = Axis(fig2[3, 2], title = "iEX"); hidespines!(ax4a)
ax4b = Axis(fig2[3, 3], title = "iEX2"); hidespines!(ax4b)
ax4c = Axis(fig2[3, 4], title = "Ca_flux"); hidespines!(ax4c)
ax4d = Axis(fig2[3, 5], title = "[Ca]s"); hidespines!(ax4d)

for (i, sol) in enumerate(data_series)

    V_t = map(t -> sol(t)[20], t_rng) #This is the voltage equation, but we are not using it in this model
    G_t = map(t -> sol(t)[4], t_rng)
    _Ca_s = map(t -> sol(t)[14], t_rng)
    O1 = map(t -> sol(t)[7], t_rng)
    O2 = map(t -> sol(t)[8], t_rng)
    O3 = map(t -> sol(t)[9], t_rng)
    mKV = map(t -> sol(t)[10], t_rng)
    hKV = map(t -> sol(t)[11], t_rng)
    mCa = map(t -> sol(t)[12], t_rng)
    mKCa = map(t -> sol(t)[13], t_rng)
    #Open parameters
    V = V_t
    G = G_t

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
    Ca_flux = -(iCa + iEX + iEX2) / (2F*V1) * 1e-6   # export µM s⁻¹

    vlines!(ax2a, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2a, t_rng, iPHOTO; opts(i, rng = rng)...)
        
    vlines!(ax2b, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2b, t_rng, iLEAK; opts(i, rng = rng)...)
    
    vlines!(ax2c, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2c, t_rng, iH; opts(i, rng = rng)...)

    vlines!(ax2d, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2d, t_rng, iKV; opts(i, rng = rng)...)

    vlines!(ax3a, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax3a, t_rng, iCa; opts(i, rng = rng)...)

    vlines!(ax3b, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax3b, t_rng, iKCa; opts(i, rng = rng)...)

    lines!(ax3c, t_rng, iCl; opts(i, rng = rng)...)

    lines!(ax4a, t_rng, iEX; opts(i, rng = rng)...)

    lines!(ax4b, t_rng, iEX2; opts(i, rng = rng)...)

    lines!(ax4c, t_rng, Ca_flux; opts(i, rng = rng)...)

    lines!(ax4d, t_rng, _Ca_s; opts(i, rng = rng)...)

    # vlines!(ax1, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax1, t_rng, V_t; opts(i, rng = rng)...)
end
fig2

#%% Show calcium buffering and calcium concentrations

fig3 = Figure(size = (1200, 600))
ax1 = Axis(fig3[1, 1], title = "[Ca]s"); hidespines!(ax1)
ax2 = Axis(fig3[1, 2], title = "[Ca]f"); hidespines!(ax2)
ax1a = Axis(fig3[1, 3], title = "[Ca]total"); hidespines!(ax1a)

ax3 = Axis(fig3[2, 1], title = "CaB_ls"); hidespines!(ax3)
ax4 = Axis(fig3[3, 1], title = "CaB_hs"); hidespines!(ax4)
ax5 = Axis(fig3[2, 2], title = "CaB_lf"); hidespines!(ax5)
ax6 = Axis(fig3[3, 2], title = "CaB_hf"); hidespines!(ax6)
for (i, sol) in enumerate(data_series)

    CaS_t = map(t -> sol(t)[14], t_rng)
    CaF_t = map(t -> sol(t)[15], t_rng)
    CaB_ls_t = map(t -> sol(t)[16], t_rng)
    CaB_hs_t = map(t -> sol(t)[17], t_rng)
    CaB_lf_t = map(t -> sol(t)[18], t_rng)
    CaB_hf_t = map(t -> sol(t)[19], t_rng)

    lines!(ax1, t_rng, CaS_t; opts(i, rng = rng)...)
    lines!(ax2, t_rng, CaF_t; opts(i, rng = rng)...)

    lines!(ax1a, t_rng, CaF_t.+CaS_t; opts(i, rng = rng)...)

    lines!(ax3, t_rng, CaB_ls_t; opts(i, rng = rng)...)
    lines!(ax4, t_rng, CaB_hs_t; opts(i, rng = rng)...)
    lines!(ax5, t_rng, CaB_lf_t; opts(i, rng = rng)...)
    lines!(ax6, t_rng, CaB_hf_t; opts(i, rng = rng)...)
end
fig3

#%%
fig4 = Figure(size = (1200, 600), title = "Electrical Equivalence Circuit")
ax1 = Axis(fig4[1, 1], title = "V_t"); hidespines!(ax1)
ax2 = Axis(fig4[1, 2], title = "iOS"); hidespines!(ax2a)
ax3 = Axis(fig4[1, 3], title = "iOSIS"); hidespines!(ax2b)
ax4 = Axis(fig4[1, 4], title = "iIS"); hidespines!(ax2c)

for (i, sol) in enumerate(data_series)

    V_t = map(t -> sol(t)[20], t_rng) #This is the voltage equation, but we are not using it in this model
    G_t = map(t -> sol(t)[4], t_rng)
    _Ca_s = map(t -> sol(t)[14], t_rng)
    O1 = map(t -> sol(t)[7], t_rng)
    O2 = map(t -> sol(t)[8], t_rng)
    O3 = map(t -> sol(t)[9], t_rng)
    mKV = map(t -> sol(t)[10], t_rng)
    hKV = map(t -> sol(t)[11], t_rng)
    mCa = map(t -> sol(t)[12], t_rng)
    mKCa = map(t -> sol(t)[13], t_rng)
    #Open parameters
    V = V_t
    G = G_t

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
    Ca_flux = -(iCa + iEX + iEX2) / (2F*V1) * 1e-6   # export µM s⁻¹

    #Map out each current according to the ERG model
    iOS = iPHOTO
    iOSIS = iLEAK
    iIS = iISCB = iCB = iAX = iST = iCa + iCl + iEX + iEX2 + iKCa + iH + iKV

    vlines!(ax1, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax1, t_rng, V_t; opts(i, rng = rng)...)

    vlines!(ax2, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2, t_rng, iOS; opts(i, rng = rng)...)
        
    vlines!(ax3, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax3, t_rng, iOSIS; opts(i, rng = rng)...)
    
    vlines!(ax4, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax4, t_rng, iIS; opts(i, rng = rng)...)
end
fig4

#%%
fig_aux = Figure(size = (1200, 600), title = "Auxillary functions")
ax1 = Axis(fig_aux[1, 1], title = "J∞"); hidespines!(ax1)
ax2 = Axis(fig_aux[1, 2], title = "C∞"); hidespines!(ax2)

GMP_rng = LinRange(0.0, 4.0, 1000)
Ca_rng = LinRange(0.0, 4.0, 1000)
J_rng = map(x -> J∞(x, kg), GMP_rng)
C_rng = map(x -> C∞(x, Cae, K_ex), Ca_rng)

lines!(ax1, GMP_rng, J_rng, color = :black, label = "J∞")
lines!(ax2, Ca_rng, C_rng, color = :black, label = "C∞")
ax1.xlabel = "GMP"
ax2.xlabel = "Ca"
ax1.ylabel = "J∞"
ax2.ylabel = "C∞"
ax1.title = "J∞"
ax2.title = "C∞"