using Revise
using DigitalTwin
import DigitalTwin: SteadyStateProblem
import DigitalTwin: J
using GLMakie, PhysiologyPlotting

using DataFrames, CSV

stim_start = 10.0
stim_end = 11.0
photons = 400.0

#%% make the model
param_df = CSV.read(raw"E:\KozLearn\Standards\full_starting_params.csv", DataFrame)
keys = param_df.Key
p0 = param_df.Value
opt_params = deepcopy(p0) # This is the parameter set that will be optimizes
lower_bounds = param_df.LowerBounds
upper_bounds = param_df.UpperBounds

u = zeros(8)
u[1] = -36.186
u[6] = 0.3 #The initial calcium value
u[7] = 34.88 #The initial cab value
u[8] = 2.0
tspan = (0.0, 1000.0)
time = LinRange(tspan[1], tspan[2], 1000)
model!(du, u, p, t) = DigitalTwin.photoreceptor_model!(du, u, p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photons)
#Need to find the steady state first
prob = ODEProblem(model!, u, tspan, p0)
steady_prob = SteadyStateProblem(prob)
sol_SS = solve(steady_prob)
sol_SS.u
sol = solve(prob, Tsit5(), tstops=[stim_start, stim_end])

fig = Figure(size = (1200, 600))
ax1 = Axis(fig[1, 1], title = "Phototransduction"); hidespines!(ax1)
ax2a = Axis(fig[2, 1], title = "Calcium dynamics"); hidespines!(ax2a)
ax2b = Axis(fig[2, 2], title = "cGMP"); hidespines!(ax2b)
ax3 = Axis(fig[3, 1], title = "Voltage"); hidespines!(ax3)

v_t = map(t -> sol(t)[1], time)
Rh_t = map(t -> sol(t)[2], time)
Rhi_t = map(t -> sol(t)[3], time)
Tr_t = map(t -> sol(t)[4], time)
PDE_t = map(t -> sol(t)[5], time)
Ca_t = map(t -> sol(t)[6], time)
CaB_t = map(t -> sol(t)[7], time)
cGMP_t = map(t -> sol(t)[8], time)

lines!(ax1, time, Rh_t, color = :blue, label = "Rho")
lines!(ax1, time, Tr_t, color = :red, label = "Tr")
lines!(ax1, time, PDE_t, color = :green, label = "PDE")

lines!(ax2a, time, Ca_t, color = :orange, label = "Ca")
#lines!(ax2a, time, CaB_t, color = :purple, label = "CaB")

lines!(ax2b, time, cGMP_t, color = :cyan, label = "cGMP")

lines!(ax3, time, v_t, color = :black, label = "V")
fig