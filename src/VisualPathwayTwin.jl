# VisualPathwayTwin
# A digital twin of the visual pathway from retina to cortex
# Based on Morris-Lecar neurons with E/I/M neurotransmitter systems
# 
# Author: Matt Tarchick
# Based on Tarchick et al. 2023 (Scientific Reports)

module VisualPathwayTwin

# Include submodules
include("Neurons.jl")
include("CellTypes.jl")
include("Networks.jl")

# Re-export key types and functions
using .Neurons
using .CellTypes
using .Networks

# Core neuron exports
export NeuronState, NeuronParams
export morris_lecar_derivatives!
export gating_steady_state, sigmoidal_release
export logistic_growth, second_messenger_dynamics

# Cell type exports
export AbstractCellType
export BipolarCell, GanglionCell
export ThalamicRelay, ThalamicInhibitory
export CorticalPyramidal, CorticalInhibitory, CorticalModulatory
export get_cell_params, CellTypeConfig
export BIPOLAR_PARAMS, GANGLION_PARAMS
export THALAMIC_RELAY_PARAMS, THALAMIC_INHIBITORY_PARAMS
export CORTICAL_PYRAMIDAL_PARAMS, CORTICAL_INHIBITORY_PARAMS, CORTICAL_MODULATORY_PARAMS

# Network exports
export NetworkLayer, VisualPathway, Synapse
export create_layer, create_visual_pathway
export connect_topographic!, connect_convergent!, connect_divergent!
export step_network!, simulate!

# High-level convenience functions

"""
    create_default_pathway(; size=:small)

Create a visual pathway with sensible defaults

Arguments:
- size: :small (8x8 retina), :medium (16x16), :large (32x32)
"""
function create_default_pathway(; size::Symbol = :medium, heterogeneous::Bool = true)
    sizes = Dict(
        :small => ((8, 8), (6, 6), (12, 12)),
        :medium => ((16, 16), (12, 12), (24, 24)),
        :large => ((32, 32), (24, 24), (48, 48))
    )
    
    retina, lgn, cortex = sizes[size]
    
    return create_visual_pathway(
        retina_size = retina,
        lgn_size = lgn,
        cortex_size = cortex,
        heterogeneous = heterogeneous
    )
end

"""
    stimulate_retina!(pathway, stimulus; duration=100.0)

Apply a stimulus pattern to bipolar cells

Arguments:
- pathway: VisualPathway
- stimulus: Matrix of applied currents (pA) matching bipolar layer dimensions
- duration: How long to apply stimulus (ms)
"""
function stimulate_retina!(
    pathway::VisualPathway{T},
    stimulus::AbstractMatrix{T};
    duration::T = T(100.0)
) where T
    bipolar_idx = pathway.layer_names["Bipolar"]
    bipolar = pathway.layers[bipolar_idx]
    
    # Apply stimulus as applied current
    for i in 1:bipolar.n_neurons
        bipolar.params[i] = NeuronParams{T}(
            bipolar.params[i];
            I_app = stimulus[i]
        )
    end
    
    # Run simulation
    times, recordings = simulate!(pathway, duration)
    
    # Remove stimulus
    for i in 1:bipolar.n_neurons
        bipolar.params[i] = NeuronParams{T}(
            bipolar.params[i];
            I_app = zero(T)
        )
    end
    
    return times, recordings
end

"""
    get_layer_activity(pathway, layer_name)

Get current voltage states for a layer as a 2D array
"""
function get_layer_activity(pathway::VisualPathway{T}, layer_name::String) where T
    idx = pathway.layer_names[layer_name]
    layer = pathway.layers[idx]
    return reshape(layer.states[:, 1], layer.n_rows, layer.n_cols)
end

"""
    set_modulatory_reversal!(pathway, layer_name, E_MOD)

Set the modulatory channel reversal potential for all neurons in a layer
This is a key tunable parameter for the E/I/M system
"""
function set_modulatory_reversal!(
    pathway::VisualPathway{T},
    layer_name::String,
    E_MOD::T
) where T
    idx = pathway.layer_names[layer_name]
    layer = pathway.layers[idx]
    
    for i in 1:layer.n_neurons
        layer.params[i] = NeuronParams{T}(
            layer.params[i];
            E_MOD = E_MOD
        )
    end
end

"""
    summary(pathway)

Print summary of pathway structure
"""
function Base.summary(pathway::VisualPathway)
    println("VisualPathwayTwin Network Summary")
    println("=" ^ 50)
    println("Current time: $(pathway.t) ms")
    println()
    println("Layers:")
    for (name, idx) in sort(collect(pathway.layer_names), by=x->x[2])
        layer = pathway.layers[idx]
        println("  $name ($(layer.cell_type))")
        println("    Neurons: $(layer.n_neurons) ($(layer.n_rows) × $(layer.n_cols))")
        
        # Count connections
        n_E_in = sum(nnz(pathway.E_connectivity[j][idx]) for j in 1:length(pathway.layers))
        n_I_in = sum(nnz(pathway.I_connectivity[j][idx]) for j in 1:length(pathway.layers))
        n_M_in = sum(nnz(pathway.M_connectivity[j][idx]) for j in 1:length(pathway.layers))
        println("    Inputs: E=$(n_E_in), I=$(n_I_in), M=$(n_M_in)")
    end
end

# Ensure submodules are accessible
const Neurons = Neurons
const CellTypes = CellTypes
const Networks = Networks

end # module VisualPathwayTwin
