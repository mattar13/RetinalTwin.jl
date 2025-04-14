using Revise
using DigitalTwin
import DigitalTwin: SteadyStateProblem
import DigitalTwin: J
using GLMakie, PhysiologyPlotting
import DigitalTwin.AutoForwardDiff
using DataFrames, CSV

#%% Input the working data
fn = raw"E:\Data\ERG\Melanopsin Data\2022_04_21_MelCreAdult\Mouse2_Adult_MelCre\NoDrugs\Rods\nd1_1p_0000.abf"
fn = raw"E:\Data\ERG\Melanopsin Data\2022_04_21_MelCreAdult\Mouse2_Adult_MelCre\BaCl_LAP4\Rods\nd1_1p_0000.abf"
fn = raw"E:\Data\ERG\Retinoschisis\2021_05_27_RS1KO-30\Mouse1_Adult_RS1KO\NoDrugs\Rods\nd1_1p_0000.abf"

dataERG, expERG = openData(fn)
stim_start = getStimulusStartTime(dataERG)[1]*1000
stim_end = getStimulusEndTime(dataERG)[1]*1000
photon_flux = 400.0
println("Stimulus runs from $(stim_start) to $(stim_end)")

#%% Set up the optimization problem
param_df = CSV.read(raw"E:\KozLearn\Standards\starting_params.csv", DataFrame)
keys = param_df.Key
p0 = param_df.Value
lower_bounds = param_df.LowerBounds
upper_bounds = param_df.UpperBounds
opt_params = deepcopy(p0) # This is the parameter set that will be optimizes

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
opt_func(p, t) = loss_static(dataERG, p; channel = 3, stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux)
opt_func(opt_params, 0.0)

# # #%% Optimize using PRIMA/COBYLA
optf = OptimizationFunction(opt_func, AutoForwardDiff())
prob = OptimizationProblem(optf, opt_params, lb = lower_bounds, ub = upper_bounds)
sol_opt = solve(prob, BOBYQA(), callback = state_callback)
opt_params = sol_opt.u

#%% Plot the ideal data
sol, ERG_t = simulate_model(dataERG, opt_params; stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux);
sol_t = dataERG.t
j_t = map(t -> sol(t)[5], sol_t)
h_t = map(t -> sol(t)[6], sol_t)
a_wave = map(t -> sol(t)[7], sol_t)
b_wave = map(t -> sol(t)[8], sol_t)
m_wave = map(t -> sol(t)[9], sol_t)
c_wave = map(t -> sol(t)[10], sol_t)
o_wave = map(t -> sol(t)[14], sol_t)
# sol_a, ERGa_t = simulate_awave(a_dataERG, opt_params);
# sol_ab, ERGab_t = simulate_abwave(ab_dataERG, opt_params);
# sol_t = abm_dataERG.t

# Start the plot for the realtime data
fig = Figure(size = (1000, 600))

ax1 = Axis(fig[1, 1], title = "BM-wave block"); hidespines!(ax1)
ax2 = Axis(fig[2, 1], title = "M-wave Block"); hidespines!(ax2)
ax3 = Axis(fig[3, 1], title = "No block");  hidespines!(ax3)
ax1b = Axis(fig[1, 2], title = "A-wave sim")
ax2b = Axis(fig[2, 2], title = "AB-wave sim");  hidespines!(ax2b)
ax3b = Axis(fig[3, 2], title = "ABM-wave sim");  hidespines!(ax3b)
ax1c = Axis(fig[1:3, 3], title = "Loss A");

# experimentplot!(ax1, dataERG, channel = 3)
# lines!(ax1, sol_t, a_wave, color = :red, label = "Simulated ERG", alpha = 0.2)
# lines!(ax1, sol_t, j_t, color = :blue, label = "Simulated ERG", alpha = 0.2)
# experimentplot!(ax2, ab_dataERG, channel = 3)
# lines!(ax2, sol_t, a_wave .+ b_wave, color = :red, label = "Simulated ERG", alpha = 0.2)
experimentplot!(ax1, dataERG, channel = 3)
lines!(ax1, sol_t, ERG_t, color = :red, label = "Simulated ERG", alpha = 0.2)

lines!(ax2b, sol_t, a_wave, color = :green, label = "Simulated ERG")
lines!(ax2b, sol_t, b_wave, color = :blue, label = "Simulated ERG")
lines!(ax2b, sol_t, a_wave .+ b_wave, color = :red, label = "Simulated ERG")

lines!(ax3b, sol_t, a_wave, color = :green, label = "Simulated ERG")
lines!(ax3b, sol_t, b_wave, color = :blue, label = "Simulated ERG")
lines!(ax3b, sol_t, m_wave, color = :magenta, label = "Simulated ERG")
lines!(ax3b, sol_t, o_wave, color = :cyan, label = "Simulated ERG")
lines!(ax3b, sol_t, c_wave, color = :orange, label = "Simulated ERG")
lines!(ax3b, sol_t, ERG_t, color = :red, label = "Simulated ERG")
fig
#%%

linkyaxes!(ax1, ax1b)
linkyaxes!(ax2, ax2b)
linkyaxes!(ax3, ax3b)

linkxaxes!(ax1, ax1b)
linkxaxes!(ax2, ax2b) 
linkxaxes!(ax3, ax3b)

fig
#%%
 # Plot the results as a vector
y = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 9, 8, 7, 6]
x = fill(1, length(y))
x[12:end] .= 3

ax4 = Axis(fig[1:2, 3], ); hidespines!(ax4); hidedecorations!(ax4)
labels = keys
pars = opt_params
marker_colors = [
    :orange, 
    :orange, 
    :blue, 
    :darkorchid, 
    :darkcyan, 
    :darkorchid, 
    :darkred, 
    :darkred, 
    :darkred,
    :darkred, 
    :darkred, 
    :orange, 
    :orange, 
    :blue, 
    :darkorchid, 
    :darkcyan,
    :darkred

]
scatter!(ax4, x, y;
    #transparency = true, 
    markersize = 52,
    marker = :circle,
    alpha = 0.6,
    color = log10.(pars),
    colormap = :viridis,
    strokecolor = marker_colors,        # dark outline
    strokewidth = 3)

for (i, lab) in enumerate(labels)
    text!(ax4, lab, position = (x[i]-0.75, y[i]), align = (:center, :center), fontsize = 20, color = :black)
    if lab == "tC"
        num = "$(round(pars[i]/10000, digits = 2))e5"
    elseif lab == "tM"
        num = "$(round(pars[i]/100, digits = 2))e3"
    else
        num = "$(round(pars[i], digits = 2))"
    end
    text!(ax4, num, position = (x[i], y[i]), align = (:center, :center), fontsize = 12, color = :black)

end
colsize!(fig.layout, 3, Relative(1/4))
xlims!(ax4, -0.0, 4.0)

fig

#%% Save the plot
save(raw"E:\KozLearn\Standards\no_drugs_rs1_trace.png", fig)

#%% Save the data if you like it
# Create a DataFrame to hold the parameter index and value
df = DataFrame(Key = keys, Value = opt_params, LowerBounds = lower_bounds, UpperBounds = upper_bounds)

# Write the DataFrame to a CSV file
CSV.write(raw"E:\KozLearn\Standards\no_drugs_params.csv", df)
#CSV.write(raw"E:\KozLearn\Standards\standard_params.csv", df)