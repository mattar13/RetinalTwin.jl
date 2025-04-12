#= 
Trying some deeper machine learning models
=#
using Pkg; Pkg.activate(".")
using Revise
using ElectroPhysiology, PhysiologyModeling
using Flux, Lux
using Random
#Pkg.activate("RetinaDeepModel")

modelA = Lux.Chain(
    Lux.Conv((3,3), 3=>1, pad = (1,1))
)
x_dist = Xoshiro()
params, state = Lux.setup(Xoshiro(), model)

function erg_ode!(du, u, p, t)
    #Extract the parameters
    dI = view(du, 1)
    dA = view(du, 2)
    dB = view(du, 3)
    dM = view(du, 4)
    dC = view(du, 5)
    dO1 = view(du, 6)
    dO2 = view(du, 7)

    #Extract the variables
    I = view(u, 1)
    A = view(u, 2)
    B = view(u, 3)
    M = view(u, 4)
    C = view(u, 5)
    O1 = view(u, 6)
    O2 = view(u, 7)

    (k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, τI, τA, τB, τM, τC) = p

    @. dI = ( k1*Stim(t) - I)/τI
    @. dA = (-k2*I^2 - A)/τA
    @. dB = (-k3*A - B)/τB
    @. dM = ( k4*A - k6*B - M)/τM
    @. dC = (-k5*A - C)/τC 
    # Oscillator equations with delayed forcing applied to the O1 system:
    @. dO1 = 0.0#O2
    # Introduce a time-dependent ramp (τ_delay set to, e.g., 0.05 sec)
    @. dO2 = 0.0#-2*k8*k10*O2 - k10^2*O1 + k7*(1 - exp(-t/k12))*B
    nothing
end

NeuralODE()

k2 = 0.1
τA = 100.0
p0 = [k2, τA, params, state]
nx, ny, nchannels = (256, 256, 3)
u0 = zeros(nx, ny, 1)
stimulus = rand(nx, ny, nchannels) 

tspan = (0.0, 1.0)
prob = ODEProblem(neural_ode, u0, tspan, p0)
sol = solve(prob, Tsit5())

sol(1.0)
sol.t
vals = cat(map(t -> sol(t)[:,:,1], sol.t)..., dims = 3)


#%%
using Random
using DiffEqFlux, Lux, Optimisers
using Pkg;Pkg.activate("test")
using GLMakie, PhysiologyPlotting
using DataFrames, CSV

#%% Open all the data
fn = raw"E:\Data\ERG\Melanopsin Data\2022_04_21_MelCreAdult\Mouse2_Adult_MelCre\BaCl_LAP4\Rods\nd1_1p_0000.abf"
dataERG = readABF(fn, flatten_episodic = false)
baseline_adjust!(dataERG, region = (0.0, 1.0), mode = :mean)
truncate_data!(dataERG, t_begin = 0.0, t_end = 4.0, truncate_based_on=:time_range)
addStimulus!(dataERG, "IN 7")
#Recalculate the stimulus protocol
stim_protocol = getStimulusProtocol(dataERG)
stim_protocol.timestamps = [(stim_protocol.timestamps[1][1], stim_protocol.timestamps[1][2])]
#Average the trials
average_trials!(dataERG)
downsample!(dataERG, 1000.0)
dataERG.t = dataERG.t*1000 #Convert this to the larger number
expERG = dataERG.data_array[1,:,1]

stim_start = getStimulusStartTime(dataERG)[1]*1000
stim_end = getStimulusEndTime(dataERG)[1]*1000
Stim(t) = stim_start <= t <= stim_end ? 400.0 : 0.0

#%% Neural network initialization
#INPUT = [St, At]
awave_model = Lux.Chain(
    Lux.Dense(2, 25, relu),
    Lux.Dense(25, 2, relu), 
)

#OUTPUT = [kA, τA]
ps, st = Lux.setup(Xoshiro(0), awave_model) #Set up the parameters with a random range
tspan = (0.0, dataERG.t[end])
function simulate_nn_model(params)
    u0 = [0.0]
    tspan = (0.0, dataERG.t[end])
    function awave_erg!(du, u, params, t)
        x = [Stim(t), u[1]]
        k, stI = awave_model(x, params, st)
        du[1] = (-k[1]*Stim(t) - u[1])/300
    end
    prob = ODEProblem(awave_erg!, u0, tspan, params) 
    sol = solve(prob, Tsit5(), saveat = dataERG.t, tstops=[stim_start, stim_end])
    return sol[1,:]
end

expERG_mat = reshape(expERG, 1, :)
# --- Define the loss function ---
# The loss function must have the signature (model, params, state, data).
function loss_func(m, ps, st, data)
    y_target = data[2]  # We expect data to be a tuple: (input, target)
    y_model = simulate_nn_model(ps)
    # Convert y_model to a 1×N matrix for comparison.
    y_model_mat = reshape(y_model, 1, :)
    return sum((y_model_mat .- y_target).^2) / length(y_model_mat)
end

# --- Set up the optimizer and TrainState using Lux's training API ---
opt = Adam(0.01)
tstate = Training.TrainState(awave_model, ps, st, opt)
vjp_rule = AutoEnzyme()

# Prepare training data as a tuple with two elements.
# For the "input" part, since your simulation doesn't depend on variable input,
# you can pass nothing (or a placeholder); the target is expERG_mat.
data = (nothing, expERG_mat)

# --- Run a single training step ---
Training.single_train_step!(vjp_rule, loss_func, data, tstate)


#%%
function train_nn_ode(tstate::Training.TrainState, vjp, data, epochs)


end

data = data |> xdev
#%% Plot the results
sol_t = dataERG.t
sol = simulate_nn_model(ps)
sol[1,:]

fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, sol_t, map(t -> Stim(t), sol_t), color = :black, label = "stim")
#Add all of the components together

ax2 = Axis(fig[2, 1])
#experimentplot!(ax2, dataERG, channel = 1)
lines!(ax2, sol_t, map(t -> sol(t)[1], sol_t), color = :red, label = "Simulated ERG")

fig