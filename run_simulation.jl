# Example: Visual Pathway Digital Twin Simulation
# 
# This script demonstrates how to:
# 1. Create a visual pathway network
# 2. Apply retinal stimulation
# 3. Record activity across all layers
# 4. Tune the modulatory channel (E_MOD parameter)
#
# Based on Tarchick et al. 2023 Morris-Lecar model

using Pkg
Pkg.activate(".")

# Load the package
include("src/VisualPathwayTwin.jl")
using .VisualPathwayTwin

using Random
using Statistics

Random.seed!(42)

println("=" ^ 60)
println("VISUAL PATHWAY DIGITAL TWIN")
println("Morris-Lecar neurons with E/I/M neurotransmitter systems")
println("=" ^ 60)
println()

#=============================================================================
    1. CREATE THE NETWORK
=============================================================================#

println("Creating visual pathway network...")
@time pathway = create_default_pathway(size=:small, heterogeneous=true)

# Print network summary
summary(pathway)
println()

#=============================================================================
    2. EXAMINE CELL TYPE PARAMETERS
=============================================================================#

println("Cell type parameter examples:")
println("-" ^ 40)

# Show key parameters for each cell type
for (name, CellType) in [
    ("Bipolar", BipolarCell),
    ("Ganglion", GanglionCell),
    ("ThalamicRelay", ThalamicRelay),
    ("CorticalPyramidal", CorticalPyramidal),
    ("CorticalInhibitory", CorticalInhibitory),
    ("CorticalModulatory", CorticalModulatory)
]
    p = get_cell_params(CellType)
    println("\n$name:")
    println("  g_Na=$(p.g_Na) nS, g_K=$(p.g_K) nS, g_Ca=$(p.g_Ca) nS")
    println("  g_E=$(p.g_E), g_I=$(p.g_I), g_M=$(p.g_M)")
    println("  E_MOD=$(p.E_MOD) mV (modulatory reversal)")
    println("  τ_A=$(p.τ_A) ms, τ_B=$(p.τ_B) ms (modulatory cascade)")
end

println()
println("=" ^ 60)

#=============================================================================
    3. RUN BASELINE SIMULATION (SPONTANEOUS ACTIVITY)
=============================================================================#

println("\nRunning baseline simulation (100 ms)...")
println("Spontaneous activity without external stimulus")

@time times_baseline, recordings_baseline = simulate!(pathway, 100.0; dt=0.1)

# Analyze spontaneous firing rates
println("\nSpontaneous firing analysis:")
println("-" ^ 40)

for (name, idx) in sort(collect(pathway.layer_names), by=x->x[2])
    voltages = recordings_baseline[idx]
    
    # Detect spikes (threshold crossing at -20 mV)
    n_neurons = size(voltages, 1)
    n_spikes = 0
    for i in 1:n_neurons
        v = voltages[i, :]
        for t in 2:length(v)
            if v[t-1] < -20.0 && v[t] >= -20.0
                n_spikes += 1
            end
        end
    end
    
    # Convert to firing rate (Hz)
    duration_s = (times_baseline[end] - times_baseline[1]) / 1000.0
    rate = n_spikes / (n_neurons * duration_s)
    
    mean_v = mean(voltages)
    std_v = std(voltages)
    
    println("$name:")
    println("  Mean voltage: $(round(mean_v, digits=1)) ± $(round(std_v, digits=1)) mV")
    println("  Firing rate: $(round(rate, digits=1)) Hz")
end

#=============================================================================
    4. SIMULATE VISUAL STIMULUS
=============================================================================#

println()
println("=" ^ 60)
println("\nSimulating visual stimulus...")

# Reset pathway time
pathway.t = 0.0

# Create a simple stimulus (bright spot in center)
bipolar_layer = pathway.layers[pathway.layer_names["Bipolar"]]
stimulus = zeros(Float64, bipolar_layer.n_rows, bipolar_layer.n_cols)

# Central bright spot
center_r, center_c = bipolar_layer.n_rows ÷ 2, bipolar_layer.n_cols ÷ 2
for dr in -2:2, dc in -2:2
    r, c = center_r + dr, center_c + dc
    if 1 <= r <= bipolar_layer.n_rows && 1 <= c <= bipolar_layer.n_cols
        dist = sqrt(dr^2 + dc^2)
        stimulus[r, c] = 15.0 * exp(-dist / 1.5)  # 15 pA peak stimulus
    end
end

println("Stimulus: Central bright spot (15 pA peak)")
println("Applied to bipolar cells at center of retina")

# Run with stimulus
stimulus_flat = vec(stimulus)
@time times_stim, recordings_stim = stimulate_retina!(pathway, reshape(stimulus_flat, :, 1)[:]; 
                                                       duration=200.0)

# Analyze evoked activity
println("\nEvoked response analysis (first 50 ms after stimulus):")
println("-" ^ 40)

for (name, idx) in sort(collect(pathway.layer_names), by=x->x[2])
    voltages = recordings_stim[idx]
    
    # Look at early response (stimulus onset)
    early_idx = findfirst(t -> t >= 50.0, times_stim)
    early_idx = early_idx === nothing ? size(voltages, 2) : early_idx
    
    early_v = voltages[:, 1:early_idx]
    
    n_neurons = size(early_v, 1)
    n_spikes = 0
    for i in 1:n_neurons
        v = early_v[i, :]
        for t in 2:length(v)
            if v[t-1] < -20.0 && v[t] >= -20.0
                n_spikes += 1
            end
        end
    end
    
    # Response latency - find first spike
    first_spike_time = Inf
    for i in 1:n_neurons
        v = voltages[i, :]
        for t in 2:length(v)
            if v[t-1] < -20.0 && v[t] >= -20.0
                first_spike_time = min(first_spike_time, times_stim[t])
                break
            end
        end
    end
    
    duration_s = times_stim[early_idx] / 1000.0
    rate = n_spikes / (n_neurons * duration_s)
    
    latency_str = first_spike_time == Inf ? "no spikes" : "$(round(first_spike_time, digits=1)) ms"
    
    println("$name: $(round(rate, digits=1)) Hz (latency: $latency_str)")
end

#=============================================================================
    5. MODULATORY PARAMETER EXPLORATION
=============================================================================#

println()
println("=" ^ 60)
println("\nModulatory channel (E_MOD) parameter exploration")
println("Testing how E_MOD affects cortical pyramidal activity")
println("-" ^ 40)

# Test different E_MOD values
E_MOD_values = [-80.0, -65.0, -50.0, -30.0]  # mV

for E_MOD in E_MOD_values
    # Create fresh pathway
    test_pathway = create_default_pathway(size=:small, heterogeneous=false)
    
    # Set E_MOD for cortical pyramidal cells
    set_modulatory_reversal!(test_pathway, "CorticalPyramidal", E_MOD)
    
    # Apply same stimulus
    test_times, test_rec = stimulate_retina!(test_pathway, reshape(stimulus_flat, :, 1)[:];
                                              duration=100.0)
    
    # Measure cortical activity
    pyr_idx = test_pathway.layer_names["CorticalPyramidal"]
    pyr_v = test_rec[pyr_idx]
    
    # Count spikes
    n_spikes = 0
    for i in 1:size(pyr_v, 1)
        v = pyr_v[i, :]
        for t in 2:length(v)
            if v[t-1] < -20.0 && v[t] >= -20.0
                n_spikes += 1
            end
        end
    end
    
    rate = n_spikes / (size(pyr_v, 1) * 0.1)  # Hz
    mean_v = mean(pyr_v)
    
    effect = E_MOD >= -50.0 ? "excitatory" : (E_MOD <= -70.0 ? "inhibitory" : "shunting")
    
    println("E_MOD = $(E_MOD) mV ($effect)")
    println("  Cortical pyramidal: $(round(rate, digits=1)) Hz, mean V = $(round(mean_v, digits=1)) mV")
end

#=============================================================================
    6. EXAMINE MODULATORY CASCADE
=============================================================================#

println()
println("=" ^ 60)
println("\nModulatory cascade analysis (At → Bt → iMod)")
println("Examining second messenger dynamics in a single neuron")
println("-" ^ 40)

# Get parameters for cortical modulatory interneuron
mod_params = get_cell_params(CorticalModulatory)

println("\nModulatory cascade parameters:")
println("  α_A (At activation rate): $(mod_params.α_A)")
println("  τ_A (At time constant): $(mod_params.τ_A) ms")
println("  β_B (Bt activation rate): $(mod_params.β_B)")
println("  τ_B (Bt time constant): $(mod_params.τ_B) ms")
println("  g_MOD (modulatory conductance): $(mod_params.g_MOD)")
println("  E_MOD (modulatory reversal): $(mod_params.E_MOD) mV")

println("\nThe cascade works as follows:")
println("  1. Modulatory NT binds receptors → activates At (logistic growth)")
println("  2. At drives second messenger Bt (4th order kinetics)")
println("  3. Bt opens/closes iMod channel with reversal E_MOD")
println("  4. E_MOD determines if modulation is excitatory, inhibitory, or shunting")

#=============================================================================
    SUMMARY
=============================================================================#

println()
println("=" ^ 60)
println("SIMULATION COMPLETE")
println("=" ^ 60)
println()
println("Key features demonstrated:")
println("  ✓ Morris-Lecar neurons with E/I/M neurotransmitter systems")
println("  ✓ Full visual pathway: Retina → LGN → V1")
println("  ✓ Cell type-specific parameters")
println("  ✓ Discrete synaptic connectivity (no PDEs)")
println("  ✓ Modulatory cascade: At → Bt → iMod with tunable E_MOD")
println("  ✓ Stimulus-evoked responses propagating through pathway")
println()
println("Next steps for optimization:")
println("  • Fit parameters to electrophysiology data")
println("  • Implement genetic algorithm for parameter optimization")
println("  • Add detailed ON/OFF pathway separation")
println("  • Implement direction selectivity circuits")
println("  • Add realistic visual input processing")
