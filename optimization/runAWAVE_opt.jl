using Revise
using DigitalTwin
using ElectroPhysiology
using Statistics
using Optimization, OptimizationBBO, OptimizationPRIMA, OptimizationOptimJL

using GLMakie, PhysiologyPlotting
using DataFrames, CSV

#%% Open the data
a1_fn = raw"E:\Data\ERG\Retinoschisis\2022_07_11_Adult\Mouse1_Adult_WT\BaCl_LAP4\Rods\nd1_1p_0000.abf"
a2_fn = raw"E:\Data\ERG\Retinoschisis\2022_07_11_Adult\Mouse1_Adult_WT\BaCl_LAP4\Rods\nd2_1p_0000.abf"
a3_fn = raw"E:\Data\ERG\Retinoschisis\2022_07_11_Adult\Mouse1_Adult_WT\BaCl_LAP4\Rods\nd3_1p_0000.abf"
a4_fn = raw"E:\Data\ERG\Retinoschisis\2022_07_11_Adult\Mouse1_Adult_WT\BaCl_LAP4\Rods\nd4_1p_0000.abf"

a1_dataERG, a1_expERG = openData(a1_fn)
a2_dataERG, a2_expERG = openData(a2_fn)
a3_dataERG, a3_expERG = openData(a3_fn)
a4_dataERG, a4_expERG = openData(a4_fn)

data_series = [a1_dataERG, a2_dataERG, a3_dataERG, a4_dataERG]
photons = [400, 40, 4, 0.4]
data_dict = Dict(zip(photons, data_series))

#%% Set up the stimulus
stim_start = getStimulusStartTime(a1_dataERG)[1]*1000
stim_end = getStimulusEndTime(a1_dataERG)[1]*1000
photon_flux = 400.0
println("Stimulus runs from $(stim_start) to $(stim_end)")

#%%Open the initial parameters
param_df = CSV.read(raw"E:\KozLearn\Standards\starting_params.csv", DataFrame)
keys = param_df.Key
p0 = param_df.Value
opt_params = deepcopy(p0) # This is the parameter set that will be optimizes
lower_bounds = param_df.LowerBounds
upper_bounds = param_df.UpperBounds

# Define the callback function
function state_callback(state, l)
    if state.iter % 25 == 1 
        println("Iteration: $(state.iter), Loss: $l")
    end
    #If we start reaching a point where the lines become flat, we should stop
    if state.iter > 10000
        return true
    else
        return false
    end
end
opt_func(p, t) = loss_graded(data_dict, p; channel = 3, stim_start = stim_start, stim_end = stim_end)[1]

# #%% Optimize using Black Box Optimization
prob = OptimizationProblem(opt_func, p0, lb = lower_bounds, ub = upper_bounds)
sol_BBO = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), callback = state_callback)
opt_params = sol_BBO.u

# # #%% Optimize using PRIMA/COBYLA
optf = OptimizationFunction(opt_func, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, opt_params, lb = lower_bounds, ub = upper_bounds)
sol_opt = solve(prob, BOBYQA(), callback = state_callback)
opt_params = sol_opt.u

#%% Run the simulation
sim = []
for (i, exp) in enumerate(data_series)
    sol_t = exp.t
    photon_flux = photons[i]
    #println("Simulating for $photon_flux")
    sol, ERG_t = simulate_model_photo(exp, opt_params, 
                        stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux
                       )
    push!(sim, sol)
    
end

#%% Make a more detailed figure
fig = Figure(size = (1200, 600))

ax1a = Axis(fig[1, 1], title = "Rhodopsin"); hidespines!(ax1a)
ax1b = Axis(fig[1, 2], title = "Transducin"); hidespines!(ax1b)

ax1c = Axis(fig[2, 1], title = "PDE"); hidespines!(ax1c)
ax1d = Axis(fig[2, 2], title = "cGMP"); hidespines!(ax1d)

ax2a = Axis(fig[3,1], title = "Jt"); hidespines!(ax2a)
ax2b = Axis(fig[3,2], title = "Ht"); hidespines!(ax2b)

ax1 = Axis(fig[1, 3], title = "Data"); hidespines!(ax1)
ax2 = Axis(fig[2,3], title = "Simulation"); hidespines!(ax2)
ax3 = Axis(fig[3,3], title = "Weighted MSE"); hidespines!(ax3)

linkyaxes!(ax1, ax2)
for (i, sol) in enumerate(sim)
    r_t = map(t -> sol(t)[1], sol.t)
    t_t = map(t -> sol(t)[2], sol.t)
    p_t = map(t -> sol(t)[3], sol.t)
    g_t = map(t -> sol(t)[4], sol.t)
    j_t = map(t -> sol(t)[5], sol.t)
    h_t = map(t -> sol(t)[6], sol.t)
    exper = data_series[i]
    
    a_wave = map(t -> sol(t)[7], sol.t)
    experimentplot!(ax1, exper, channel = 3)
    lines!(ax2, sol.t, a_wave, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Simulated ERG"
    )
    mse = (a_wave .- exper[1,:,3]).^2
    weights = map(t -> exp(-t/200), exper.t)  # Exponential decay weight
    weighted_loss = (weights .* mse)
    lines!(ax3, exper.t, weighted_loss, 
        color = round(log10(photons[i])), colormap = :viridis, 
        colorrange = (0.0, 4.0), label = "Loss A"
    )
    
    lines!(ax1a, sol.t, r_t, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Ht"
    )

    lines!(ax1b, sol.t, t_t, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Ht"
    )

    lines!(ax1c, sol.t, p_t, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Ht"
    )

    lines!(ax1d, sol.t, g_t, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Ht"
    )

    lines!(ax2a, sol.t, j_t, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Ht"
    )

    lines!(ax2b, sol.t, h_t, 
            color = round(log10(photons[i])), colormap = :viridis, 
            colorrange = (0.0, 4.0),
            label = "Ht"
    )

end
fig

#%% Plot the data
fig = Figure(size = (1200, 600))

ax1a = Axis(fig[1, 1], title = "A-wave series"); hidespines!(ax1a)
ax1b = Axis(fig[2, 1], title = "A-wave sim"); hidespines!(ax1b)
ax1c = Axis(fig[3, 1], title = "Loss A"); hidespines!(ax1c)
linkyaxes!(ax1a, ax1b)
for (i, sol) in enumerate(sim)
    exper = data_series[i]
    experimentplot!(ax1a, exper, channel = 3,          
        #color = round(log10(photons[i])), colormap = :viridis, 
        #colorrange = (0.0, 4.0), label = "Experimental ERG"
    )
    photon_flux = photons[i]
    println("Simulating for $photon_flux")
    sol, ERG_t = simulate_model_photo(exper, opt_params[1:16], stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux)
    a_wave = map(t -> sol(t)[7], exper.t)
    lines!(ax1b, exper.t, a_wave, 			   
        color = round(log10(photons[i])), colormap = :viridis, 
        colorrange = (0.0, 4.0), label = "Simulated ERG"
    )
    mse = (ERG_t .- exper.data_array[1,:,3]).^2

    weights = map(t -> exp(-t/200), exper.t)  # Exponential decay weight
    weighted_loss = (weights .* mse)
    lines!(ax1c, exper.t, weighted_loss, 
        color = round(log10(photons[i])), colormap = :viridis, 
        colorrange = (0.0, 4.0), label = "Loss A"
    )
end
fig

#%% Save the data if you like it
# Create a DataFrame to hold the parameter index and value
df = DataFrame(Key = keys, Value = opt_params, LowerBounds = lower_bounds, UpperBounds = upper_bounds)

# Write the DataFrame to a CSV file
CSV.write(raw"E:\KozLearn\Standards\starting_params.csv", df)
#CSV.write(raw"E:\KozLearn\Standards\standard_params.csv", df)