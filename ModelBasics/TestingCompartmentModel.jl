using Revise
using DigitalTwin
using GLMakie, PhysiologyPlotting
import DigitalTwin: Rodas5, SSRootfind, DynamicSS
import DigitalTwin.phototransduction_ode!
using DataFrames, CSV

# make the model
param_df = CSV.read(raw"Parameters\starting_params.csv", DataFrame)
keys = param_df.Key
p0 = param_df.Value
opt_params = deepcopy(p0) # This is the parameter set that will be optimizes
lower_bounds = param_df.LowerBounds
upper_bounds = param_df.UpperBounds
#stim_range = collect(-65:5:-20) #V hold
stim_range = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

cond_df = CSV.read(raw"Parameters\starting_conditions.csv", DataFrame)
u0 = cond_df.value
u0 = [u0[1:end-1] ; fill(u0[end], 7); zeros(7)]

(aC, kR1, kF1, kR2, kR3, kHYDRO, kREC, G0, iDARK, kg, 
C_m, gLEAK, eLEAK, gH, eH, gKV, eK, gCa, eCa, _Ca_0, gKCa, gCl, eCl, 
F, DCa, S1, DELTA, V1, V2, Lb1, Bl, Lb2, Hb1, Bh, Hb2,
J_ex, Cae, K_ex, J_ex2, K_ex2,
) = p0

data_series = []
stim_start = 0.100
stim_end = 0.120
# stim_start = stim_end = 0.0
tspan = (0.0, 10.0)
t_rng = LinRange(0.0, tspan[end], 1000)
for i in stim_range
    println("Simulating for $i")
    photons = i

    model!(du, u, p, t) = DigitalTwin.phototransduction_compartments!(du, u, p, t; stim_start = stim_start, stim_end = stim_end, photon_flux = photons)
    prob = ODEProblem(model!, u0, tspan, p0[1:40])
    sol = solve(prob, Rodas5(), dt = 0.001, tstops=[stim_start, stim_end])
    push!(data_series, sol)
end

# include("PlottingCompartmentModel.jl")