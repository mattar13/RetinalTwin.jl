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
ax1 = Axis(fig1[1, 1], title = "V_OS"); hidespines!(ax1a)
ax2 = Axis(fig1[1, 2], title = "V_OSIS"); hidespines!(ax1b)
ax3 = Axis(fig1[2, 1], title = "V_IS"); hidespines!(ax2a)
ax4 = Axis(fig1[2, 2], title = "V_ISCB"); hidespines!(ax2b)
ax5 = Axis(fig1[3, 1], title = "V_CB"); hidespines!(ax3)
ax6 = Axis(fig1[3, 2], title = "V_AX"); hidespines!(ax3)
ax7 = Axis(fig1[4, 1], title = "V_ST"); hidespines!(ax3)

for (i, sol) in enumerate(data_series)
    Vos_t = map(t -> sol(t)[20], t_rng) #This is the voltage equation, but we are not using it in this model
    Vosis_t = map(t -> sol(t)[21], t_rng) #This is the voltage equation, but we are not using it in this model
    Vis_t = map(t -> sol(t)[22], t_rng) #This is the voltage equation, but we are not using it in this model
    Viscb_t = map(t -> sol(t)[23], t_rng) #This is the voltage equation, but we are not using it in this model
    Vcb_t = map(t -> sol(t)[24], t_rng) #This is the voltage equation, but we are not using it in this model
    Vax_t = map(t -> sol(t)[25], t_rng) #This is the voltage equation, but we are not using it in this model
    Vst_t = map(t -> sol(t)[26], t_rng) #This is the voltage equation, but we are not using it in this model

    vlines!(ax1, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax1, t_rng, Vos_t; opts(i, rng = rng)...)

    vlines!(ax2, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax2, t_rng, Vosis_t; opts(i, rng = rng)...)

    vlines!(ax3, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax3, t_rng, Vis_t; opts(i, rng = rng)...)

    vlines!(ax4, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax4, t_rng, Viscb_t; opts(i, rng = rng)...)

    vlines!(ax5, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax5, t_rng, Vcb_t; opts(i, rng = rng)...)

    vlines!(ax6, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax6, t_rng, Vax_t; opts(i, rng = rng)...)

    vlines!(ax7, [stim_start, stim_end], (-0.5, 1.0), color = :black, alpha = 0.2)
    lines!(ax7, t_rng, Vst_t; opts(i, rng = rng)...)
end
fig1