using Revise
using DigitalTwin
import DigitalTwin: SteadyStateProblem
import DigitalTwin: J
using GLMakie, PhysiologyPlotting
import DigitalTwin.AutoForwardDiff
using DataFrames, CSV

#%% Open the data
abm_fn = raw"E:\Data\ERG\Melanopsin Data\2022_04_21_MelCreAdult\Mouse2_Adult_MelCre\NoDrugs\Rods\nd1_1p_0000.abf"
ab_fn = raw"E:\Data\ERG\Melanopsin Data\2022_04_21_MelCreAdult\Mouse2_Adult_MelCre\BaCl\Rods\nd1_1p_0000.abf"
a_fn = raw"E:\Data\ERG\Melanopsin Data\2022_04_21_MelCreAdult\Mouse2_Adult_MelCre\BaCl_LAP4\Rods\nd1_1p_0000.abf"

abm_dataERG, abm_expERG = openData(abm_fn)
ab_dataERG, ab_expERG = openData(ab_fn)
a_dataERG, a_expERG = openData(a_fn)

#Set up the stimulus
stim_start = getStimulusStartTime(abm_dataERG)[1]*1000
stim_end = getStimulusEndTime(abm_dataERG)[1]*1000
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
    if state.iter > 80000
        return true
    else
        return false
    end
end
using Statistics
opt_func_full(p, t) = loss_static_abm(a_dataERG, ab_dataERG, abm_dataERG, p; 
    stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux
)[1] |> sum
#%% Optimize using Black Box Optimization
# prob = OptimizationProblem(opt_func_full, p0, lb = lower_bounds, ub = upper_bounds)
# sol_BBO = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), callback = state_callback)
# opt_params = sol_BBO.u
# opt_func_full(opt_params, 0.0)

#%% Optimize using PRIMA/COBYLA
optf = OptimizationFunction(opt_func_full, AutoForwardDiff())
prob = OptimizationProblem(optf, opt_params, lb = lower_bounds, ub = upper_bounds)

sol_opt = solve(prob, BOBYQA(), callback = state_callback)
opt_params = sol_opt.u
opt_func_full(opt_params, 0.0)

#%% Plot the ideal data
sol, ERG_t = simulate_model(abm_dataERG, opt_params; stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux);
sol_t = abm_dataERG.t
j_t = map(t -> sol(t)[5], sol_t)
h_t = map(t -> sol(t)[6], sol_t)
a_wave = map(t -> sol(t)[7], sol_t)
b_wave = map(t -> sol(t)[8], sol_t)
m_wave = map(t -> sol(t)[9], sol_t)
c_wave = map(t -> sol(t)[10], sol_t)
o_wave = map(t -> sol(t)[14], sol_t)

# Start the plot for the realtime data
fig = Figure(size = (1000, 600))

ax1 = Axis(fig[1, 1], title = "BM-wave block"); hidespines!(ax1)
ax2 = Axis(fig[2, 1], title = "M-wave Block"); hidespines!(ax2)
ax3 = Axis(fig[3, 1], title = "No block"); hidespines!(ax3)
ax1b = Axis(fig[1, 2], title = "A-wave sim"); hidespines!(ax1b)
ax2b = Axis(fig[2, 2], title = "AB-wave sim"); hidespines!(ax2b)
ax3b = Axis(fig[3, 2], title = "ABM-wave sim"); hidespines!(ax3b)
ax1c = Axis(fig[1:3, 3], title = "Loss A");

experimentplot!(ax1, a_dataERG, channel = 3)
lines!(ax1, sol_t, a_wave, color = :red, label = "Simulated ERG", alpha = 0.2)

experimentplot!(ax2, ab_dataERG, channel = 3)
lines!(ax2, sol_t, a_wave .+ b_wave, color = :red, label = "Simulated ERG", alpha = 0.2)

experimentplot!(ax3, abm_dataERG, channel = 3)
lines!(ax3, sol_t, ERG_t, color = :red, label = "Simulated ERG", alpha = 0.2)

lines!(ax1b, sol_t, a_wave, color = :red, label = "Simulated ERG")

lines!(ax2b, sol_t, a_wave, color = :green, label = "Simulated ERG")
lines!(ax2b, sol_t, b_wave, color = :blue, label = "Simulated ERG")
lines!(ax2b, sol_t, a_wave .+ b_wave, color = :red, label = "Simulated ERG")

lines!(ax3b, sol_t, a_wave, color = :green, label = "Simulated ERG")
lines!(ax3b, sol_t, b_wave, color = :blue, label = "Simulated ERG")
lines!(ax3b, sol_t, m_wave, color = :magenta, label = "Simulated ERG")
lines!(ax3b, sol_t, o_wave, color = :cyan, label = "Simulated ERG")
lines!(ax3b, sol_t, c_wave, color = :orange, label = "Simulated ERG")
lines!(ax3b, sol_t, ERG_t, color = :red, label = "Simulated ERG")

linkyaxes!(ax1, ax1b)
linkyaxes!(ax2, ax2b)
linkyaxes!(ax3, ax3b)

linkxaxes!(ax1, ax1b)
linkxaxes!(ax2, ax2b) 
linkxaxes!(ax3, ax3b)

fig

#%% Save the data if you like it
 # Create a DataFrame to hold the parameter index and value
df = DataFrame(Key = keys, Value = opt_params, LowerBounds = lower_bounds, UpperBounds = upper_bounds)

# Write the DataFrame to a CSV file
CSV.write(raw"E:\KozLearn\Standards\starting_params.csv", df)
#CSV.write(raw"E:\KozLearn\Standards\standard_params.csv", df)