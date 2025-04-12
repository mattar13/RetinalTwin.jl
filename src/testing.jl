using Revise
using DigitalTwin
import DigitalTwin: SteadyStateProblem
import DigitalTwin: J
using GLMakie, PhysiologyPlotting

using DataFrames, CSV

#%% make the model
param_df = CSV.read(raw"E:\KozLearn\Standards\phototrans_params.csv", DataFrame)
keys = param_df.Key
p0 = param_df.Value
opt_params = deepcopy(p0) # This is the parameter set that will be optimizes
lower_bounds = param_df.LowerBounds
upper_bounds = param_df.UpperBounds
photon_range = [0.4, 4.0, 40.0, 400.0, 4000.0]

data_series = []
stim_start = 10.0
stim_end = 11.0
for i in photon_range
    println("Simulating for $i")
    photons = i

    u = zeros(4)
    tspan = (0.0, 5000.0)
    time = LinRange(tspan[1], tspan[2], 1000)
    model!(du, u, p, t) = DigitalTwin.phototransduction_model!(du, u, p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photons)
    prob = ODEProblem(model!, u, tspan, p0)
    sol = solve(prob, Tsit5(), tstops=[stim_start, stim_end])
    push!(data_series, sol)
end
#%%
time = LinRange(tspan[1], tspan[2], 1000)
model!(du, u, p, t) = DigitalTwin.phototransduction_model!(du, u, p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photons)
prob = ODEProblem(model!, u, tspan, p0)
sol = solve(prob, Tsit5(), tstops=[stim_start, stim_end])

#%%
push!(data_series, sol)
fig = Figure(size = (1200, 600))
ax1 = Axis(fig[1, 1], title = "Phototransduction"); hidespines!(ax1)
ax2a = Axis(fig[2, 1], title = "Calcium dynamics"); hidespines!(ax2a)
ax2b = Axis(fig[2, 2], title = "cGMP"); hidespines!(ax2b)
ax3 = Axis(fig[3, 1], title = "Voltage"); hidespines!(ax3)

for (i, sol) in enumerate(data_series)
    Rh_t = map(t -> sol(t)[1], time)
    Tr_t = map(t -> sol(t)[2], time)
    PDE_t = map(t -> sol(t)[3], time)
    C_t = map(t -> sol(t)[4], time)

    # lines!(ax1, time, Rh_t,             
    #     color = round(log10(photon_range[i])), colormap = :viridis, 
    #     colorrange = (0.0, 3.0),
    #     label = "Rho"
    # )
    lines!(ax1, time, Tr_t,         
        color = round(log10(photon_range[i])), colormap = :viridis, 
        colorrange = (0.0, 3.0),
        label = "Tr")
    # lines!(ax1, time, PDE_t, color = :green, label = "PDE")

    # lines!(ax2a, time, Ca_t,         
    #     color = round(log10(photon_range[i])), colormap = :viridis, 
    #     colorrange = (0.0, 3.0), label = "Ca")
    # #lines!(ax2a, time, CaB_t, color = :purple, label = "CaB")

    # lines!(ax2b, time, cGMP_t,         
    #     color = round(log10(photon_range[i])), colormap = :viridis, 
    #     colorrange = (0.0, 3.0),label = "cGMP")

    # lines!(ax3, time, v_t,         
    #     color = round(log10(photon_range[i])), colormap = :viridis, 
    #     colorrange = (0.0, 3.0), label = "V")
end
fig