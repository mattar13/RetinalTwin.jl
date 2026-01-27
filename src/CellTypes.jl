# Cell-type specific parameter presets for the visual pathway
# Each cell type has optimized parameters based on known physiology

module CellTypes

using ..Neurons
using Parameters

export get_cell_params, CellTypeConfig
export BIPOLAR_PARAMS, GANGLION_PARAMS, THALAMIC_RELAY_PARAMS
export THALAMIC_INHIBITORY_PARAMS, CORTICAL_PYRAMIDAL_PARAMS
export CORTICAL_INHIBITORY_PARAMS, CORTICAL_MODULATORY_PARAMS

#=============================================================================
    CELL TYPE CONFIGURATION
    
    Contains metadata about each cell type:
    - Primary neurotransmitter released (E, I, or M)
    - Typical input sources
    - Layer/region information
=============================================================================#

@with_kw struct CellTypeConfig
    name::String
    region::String                    # "retina", "lgn", "v1"
    layer::String                     # Sublayer info
    primary_nt::Symbol                # :excitatory, :inhibitory, :modulatory
    releases_E::Bool = false          # Releases excitatory NT
    releases_I::Bool = false          # Releases inhibitory NT  
    releases_M::Bool = false          # Releases modulatory NT
    receives_from::Vector{String} = String[]  # Cell types providing input
end

#=============================================================================
    BIPOLAR CELLS (Retina)
    
    - ON and OFF subtypes (parameterized by E_Cl for GABA response)
    - Receive glutamate from photoreceptors
    - Release glutamate to ganglion cells and amacrines
    - Graded potentials, minimal spiking
=============================================================================#

const BIPOLAR_CONFIG = CellTypeConfig(
    name = "Bipolar",
    region = "retina",
    layer = "INL",
    primary_nt = :excitatory,
    releases_E = true,
    receives_from = ["Photoreceptor"]
)

const BIPOLAR_PARAMS = NeuronParams(
    # Lower capacitance - smaller cells
    C_m = 8.0,
    
    # Graded response characteristics
    g_leak = 3.0,
    E_leak = -55.0,      # More depolarized resting
    
    # Calcium channels - key for graded transmission
    g_Ca = 6.0,          # Strong Ca for synaptic release
    E_Ca = 120.0,
    V1 = -25.0,          # More depolarized activation
    V2 = 12.0,
    
    # Minimal K+ for graded response
    g_K = 4.0,
    E_K = -90.0,
    V3 = -5.0,
    V4 = 15.0,
    τ_N = 8.0,
    
    # Minimal Na+ (non-spiking)
    g_Na = 2.0,
    E_Na = 50.0,
    
    # Modulatory channel - tunable
    g_MOD = 0.2,
    E_MOD = -60.0,       # Slightly inhibitory modulation
    
    # Strong excitatory output (glutamate)
    g_E = 0.3,
    E_E = 0.0,
    ρ_E = 8.0,           # Strong glutamate release
    τ_E = 5.0,           # Fast glutamate kinetics
    
    # No inhibitory output
    g_I = 0.0,
    ρ_I = 0.0,
    
    # Weak modulatory
    g_M = 0.05,
    ρ_M = 1.0,
    
    # Low noise - stable graded responses
    σ_noise = 0.05,
    τ_W = 500.0
)

#=============================================================================
    RETINAL GANGLION CELLS (RGCs)
    
    - Spiking neurons
    - Receive glutamate from bipolars, GABA from amacrines
    - Project to LGN via optic nerve
    - Multiple types: ON, OFF, ON-OFF, direction-selective, etc.
=============================================================================#

const GANGLION_CONFIG = CellTypeConfig(
    name = "Ganglion",
    region = "retina",
    layer = "GCL",
    primary_nt = :excitatory,
    releases_E = true,
    receives_from = ["Bipolar", "Amacrine"]
)

const GANGLION_PARAMS = NeuronParams(
    # Standard RGC membrane properties
    C_m = 15.0,
    
    g_leak = 2.5,
    E_leak = -65.0,
    
    # Calcium channels
    g_Ca = 4.0,
    E_Ca = 120.0,
    V1 = -1.0,
    V2 = 18.0,
    
    # Strong K+ for repolarization
    g_K = 10.0,
    E_K = -90.0,
    V3 = 10.0,
    V4 = 15.0,
    τ_N = 5.0,
    
    # Strong Na+ for action potentials
    g_Na = 35.0,
    E_Na = 55.0,
    
    # TREK1 for sAHP
    g_TREK = 0.8,
    
    # Modulatory channel
    g_MOD = 0.3,
    E_MOD = -65.0,
    
    # Excitatory release (glutamate to LGN)
    g_E = 0.25,
    E_E = 0.0,
    ρ_E = 7.0,
    τ_E = 3.0,           # Fast
    
    # RGCs don't release GABA
    g_I = 0.0,
    ρ_I = 0.0,
    
    # Some modulatory release
    g_M = 0.1,
    ρ_M = 2.0,
    
    # Moderate spontaneous activity
    σ_noise = 0.15,
    τ_W = 600.0
)

#=============================================================================
    THALAMIC RELAY NEURONS (LGN)
    
    - Receive input from RGCs and cortical feedback
    - Relay visual information to V1
    - Two firing modes: tonic and burst (T-type Ca2+ dependent)
    - Strong feedback from cortex
=============================================================================#

const THALAMIC_RELAY_CONFIG = CellTypeConfig(
    name = "ThalamicRelay",
    region = "lgn",
    layer = "relay",
    primary_nt = :excitatory,
    releases_E = true,
    receives_from = ["Ganglion", "CorticalPyramidal", "ThalamicInhibitory"]
)

const THALAMIC_RELAY_PARAMS = NeuronParams(
    # Larger cells
    C_m = 20.0,
    
    g_leak = 2.0,
    E_leak = -70.0,
    
    # Strong T-type calcium for burst firing
    g_Ca = 5.5,
    E_Ca = 120.0,
    V1 = -52.0,          # Low-threshold for T-type
    V2 = 7.4,            # Steep activation
    
    # K+ channels including Kv3 for fast repolarization
    g_K = 12.0,
    E_K = -95.0,
    V3 = -25.0,
    V4 = 12.0,
    τ_N = 4.0,
    
    # Na+ for action potentials
    g_Na = 40.0,
    E_Na = 55.0,
    
    # TREK1 for afterhyperpolarization
    g_TREK = 1.0,
    
    # Modulatory - state dependent (sleep/wake via ACh/NE)
    g_MOD = 0.5,
    E_MOD = -60.0,
    
    # Strong excitatory output to cortex
    g_E = 0.35,
    E_E = 0.0,
    ρ_E = 10.0,
    τ_E = 2.0,
    
    # No inhibitory release
    g_I = 0.0,
    ρ_I = 0.0,
    
    # Modulatory to local interneurons
    g_M = 0.15,
    ρ_M = 3.0,
    
    σ_noise = 0.1,
    τ_W = 700.0
)

#=============================================================================
    THALAMIC INHIBITORY INTERNEURONS (LGN)
    
    - Local GABAergic interneurons
    - Receive input from relay cells and cortex
    - Provide feedforward and feedback inhibition
    - Important for gain control
=============================================================================#

const THALAMIC_INHIBITORY_CONFIG = CellTypeConfig(
    name = "ThalamicInhibitory",
    region = "lgn",
    layer = "interneuron",
    primary_nt = :inhibitory,
    releases_I = true,
    receives_from = ["Ganglion", "ThalamicRelay", "CorticalPyramidal"]
)

const THALAMIC_INHIBITORY_PARAMS = NeuronParams(
    # Smaller interneurons
    C_m = 12.0,
    
    g_leak = 2.5,
    E_leak = -65.0,
    
    # Moderate calcium
    g_Ca = 3.5,
    E_Ca = 120.0,
    V1 = -5.0,
    V2 = 15.0,
    
    # Fast K+ for rapid firing
    g_K = 15.0,
    E_K = -90.0,
    V3 = -10.0,
    V4 = 10.0,
    τ_N = 3.0,           # Fast gating
    
    # Na+ channels
    g_Na = 30.0,
    E_Na = 55.0,
    
    # Moderate sAHP
    g_TREK = 0.5,
    
    # Modulatory channel
    g_MOD = 0.25,
    E_MOD = -70.0,
    
    # No excitatory release
    g_E = 0.0,
    ρ_E = 0.0,
    
    # Strong GABA release
    g_I = 1.2,
    E_Cl = -80.0,        # Hyperpolarizing GABA
    ρ_I = 8.0,
    τ_I = 10.0,
    
    # Some modulatory
    g_M = 0.1,
    ρ_M = 2.0,
    
    σ_noise = 0.12,
    τ_W = 500.0
)

#=============================================================================
    CORTICAL PYRAMIDAL CELLS (V1)
    
    - Primary excitatory neurons in cortex
    - Complex dendritic morphology (simplified here)
    - Strong adaptation via Ca-activated K+ channels
    - Receive thalamic and local cortical inputs
=============================================================================#

const CORTICAL_PYRAMIDAL_CONFIG = CellTypeConfig(
    name = "CorticalPyramidal",
    region = "v1",
    layer = "2/3, 4, 5, 6",
    primary_nt = :excitatory,
    releases_E = true,
    receives_from = ["ThalamicRelay", "CorticalPyramidal", "CorticalInhibitory", "CorticalModulatory"]
)

const CORTICAL_PYRAMIDAL_PARAMS = NeuronParams(
    # Large cells with complex morphology
    C_m = 25.0,
    
    g_leak = 1.8,
    E_leak = -70.0,
    
    # Calcium channels
    g_Ca = 4.0,
    E_Ca = 120.0,
    V1 = 0.0,
    V2 = 20.0,
    
    # K+ channels - multiple types for adaptation
    g_K = 10.0,
    E_K = -90.0,
    V3 = 15.0,
    V4 = 18.0,
    τ_N = 6.0,
    
    # Strong Na+ for action potentials
    g_Na = 45.0,
    E_Na = 55.0,
    
    # Strong TREK1 for adaptation (sAHP)
    g_TREK = 1.2,
    
    # Modulatory channel - important for attention/arousal
    g_MOD = 0.4,
    E_MOD = -50.0,       # Depolarizing modulation (e.g., ACh effect)
    
    # Strong calcium dynamics for adaptation
    τ_A = 10000.0,
    τ_B = 12000.0,
    
    # Excitatory output (glutamate)
    g_E = 0.3,
    E_E = 0.0,
    ρ_E = 6.0,
    τ_E = 5.0,
    
    # No GABA release
    g_I = 0.0,
    ρ_I = 0.0,
    
    # Some modulatory release (feedback)
    g_M = 0.15,
    ρ_M = 2.5,
    
    σ_noise = 0.08,
    τ_W = 800.0
)

#=============================================================================
    CORTICAL INHIBITORY INTERNEURONS (V1)
    
    - Fast-spiking basket cells, chandelier cells
    - Parvalbumin+ (PV) type: fast, no adaptation
    - Provide perisomatic inhibition to pyramidal cells
=============================================================================#

const CORTICAL_INHIBITORY_CONFIG = CellTypeConfig(
    name = "CorticalInhibitory",
    region = "v1",
    layer = "all",
    primary_nt = :inhibitory,
    releases_I = true,
    receives_from = ["ThalamicRelay", "CorticalPyramidal", "CorticalModulatory"]
)

const CORTICAL_INHIBITORY_PARAMS = NeuronParams(
    # Smaller, compact cells
    C_m = 10.0,
    
    g_leak = 3.0,
    E_leak = -65.0,
    
    # Lower calcium
    g_Ca = 2.5,
    E_Ca = 120.0,
    V1 = -10.0,
    V2 = 12.0,
    
    # Very fast K+ (Kv3 type) for fast spiking
    g_K = 20.0,
    E_K = -85.0,
    V3 = -15.0,
    V4 = 8.0,
    τ_N = 1.5,           # Very fast
    
    # Fast Na+ channels
    g_Na = 50.0,
    E_Na = 55.0,
    
    # Low sAHP - minimal adaptation
    g_TREK = 0.2,
    
    # Weak modulatory channel
    g_MOD = 0.15,
    E_MOD = -65.0,
    
    # Shorter time constants - fast dynamics
    τ_A = 5000.0,
    τ_B = 6000.0,
    
    # No excitatory release
    g_E = 0.0,
    ρ_E = 0.0,
    
    # Strong, fast GABA release
    g_I = 1.5,
    E_Cl = -75.0,
    ρ_I = 10.0,
    τ_I = 5.0,           # Fast GABA kinetics
    
    # Minimal modulatory
    g_M = 0.05,
    ρ_M = 1.0,
    
    σ_noise = 0.1,
    τ_W = 400.0
)

#=============================================================================
    CORTICAL MODULATORY INTERNEURONS (V1)
    
    - SOM+ (somatostatin) and VIP+ interneurons
    - Target dendrites, other interneurons
    - Release GABA but also neuropeptides (modulatory)
    - Important for disinhibition circuits
=============================================================================#

const CORTICAL_MODULATORY_CONFIG = CellTypeConfig(
    name = "CorticalModulatory",
    region = "v1",
    layer = "all",
    primary_nt = :modulatory,
    releases_I = true,
    releases_M = true,
    receives_from = ["CorticalPyramidal", "CorticalInhibitory"]
)

const CORTICAL_MODULATORY_PARAMS = NeuronParams(
    # Medium-sized interneurons
    C_m = 12.0,
    
    g_leak = 2.2,
    E_leak = -60.0,      # Slightly more depolarized
    
    # Moderate calcium for peptide release
    g_Ca = 4.0,
    E_Ca = 120.0,
    V1 = -8.0,
    V2 = 16.0,
    
    # Moderate K+
    g_K = 8.0,
    E_K = -88.0,
    V3 = 5.0,
    V4 = 14.0,
    τ_N = 5.0,
    
    # Na+ channels
    g_Na = 30.0,
    E_Na = 55.0,
    
    # Moderate sAHP
    g_TREK = 0.6,
    
    # Strong modulatory channel (self-modulation)
    g_MOD = 0.5,
    E_MOD = -55.0,       # Slightly depolarizing
    
    # Longer time constants for modulatory effects
    τ_A = 15000.0,
    τ_B = 20000.0,
    
    # No excitatory release
    g_E = 0.0,
    ρ_E = 0.0,
    
    # Moderate GABA release (dendritic targeting)
    g_I = 0.8,
    E_Cl = -70.0,
    ρ_I = 5.0,
    τ_I = 20.0,          # Slower GABA kinetics
    
    # Strong modulatory release (neuropeptides: SST, VIP)
    g_M = 0.3,
    ρ_M = 5.0,
    τ_M = 200.0,         # Very slow modulatory effects
    
    σ_noise = 0.1,
    τ_W = 600.0
)

#=============================================================================
    PARAMETER ACCESS FUNCTIONS
=============================================================================#

"""
    get_cell_params(cell_type::Type{<:AbstractCellType})

Get default parameters for a specific cell type
"""
function get_cell_params(::Type{BipolarCell})
    return BIPOLAR_PARAMS
end

function get_cell_params(::Type{GanglionCell})
    return GANGLION_PARAMS
end

function get_cell_params(::Type{ThalamicRelay})
    return THALAMIC_RELAY_PARAMS
end

function get_cell_params(::Type{ThalamicInhibitory})
    return THALAMIC_INHIBITORY_PARAMS
end

function get_cell_params(::Type{CorticalPyramidal})
    return CORTICAL_PYRAMIDAL_PARAMS
end

function get_cell_params(::Type{CorticalInhibitory})
    return CORTICAL_INHIBITORY_PARAMS
end

function get_cell_params(::Type{CorticalModulatory})
    return CORTICAL_MODULATORY_PARAMS
end

"""
    get_cell_config(cell_type::Type{<:AbstractCellType})

Get configuration metadata for a cell type
"""
function get_cell_config(::Type{BipolarCell})
    return BIPOLAR_CONFIG
end

function get_cell_config(::Type{GanglionCell})
    return GANGLION_CONFIG
end

function get_cell_config(::Type{ThalamicRelay})
    return THALAMIC_RELAY_CONFIG
end

function get_cell_config(::Type{ThalamicInhibitory})
    return THALAMIC_INHIBITORY_CONFIG
end

function get_cell_config(::Type{CorticalPyramidal})
    return CORTICAL_PYRAMIDAL_CONFIG
end

function get_cell_config(::Type{CorticalInhibitory})
    return CORTICAL_INHIBITORY_CONFIG
end

function get_cell_config(::Type{CorticalModulatory})
    return CORTICAL_MODULATORY_CONFIG
end

end # module CellTypes
