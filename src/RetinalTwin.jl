module RetinalTwin

using LinearAlgebra
using Statistics

# --- Core types ---
include("types.jl")

# --- Default parameters ---
include("parameters.jl")

# --- Stimulus ---
include("stimulus/light.jl")

# --- Cell update functions ---
include("cells/photoreceptor.jl")
include("cells/horizontal.jl")
include("cells/on_bipolar.jl")
include("cells/off_bipolar.jl")
include("cells/a2_amacrine.jl")
include("cells/gaba_amacrine.jl")
include("cells/da_amacrine.jl")
include("cells/ganglion.jl")
include("cells/muller.jl")
include("cells/rpe.jl")

# --- Synaptic framework ---
include("synapses/synapse.jl")
include("synapses/mglur6.jl")
include("synapses/modulatory.jl")

# --- Circuit wiring ---
include("circuit/connectivity.jl")
include("circuit/retinal_column.jl")

# --- ERG ---
include("erg/field_potential.jl")

# --- Simulation ---
include("simulation/ode_system.jl")
include("simulation/run.jl")

# --- Visualization ---
include("visualization/plots.jl")
include("visualization/gl_plots.jl")

# --- Validation ---
include("validation/targets.jl")

# --- Public API ---
export
    # Types
    MLParams, PhototransductionParams, NTReleaseParams,
    SynapseParams, mGluR6Params, MullerParams, RPEParams,
    ERGWeights, StimulusProtocol, PopulationSizes,
    RetinalColumn, StateIndex, ConnectionDef,
    # Parameter factories
    build_retinal_column,
    default_rod_params, default_cone_params,
    default_hc_params, default_on_bc_params, default_off_bc_params,
    default_a2_params, default_gaba_params, default_da_params, default_gc_params,
    default_muller_params, default_rpe_params, default_mglur6_params,
    default_erg_weights, default_connections,
    # Stimulus
    compute_stimulus, flash_stimulus,
    # Simulation
    simulate_flash, extract_voltages, extract_neurotransmitters,
    dark_adapted_state, retinal_column_rhs!,
    # ERG
    compute_erg, extract_ops,
    # Visualization
    plot_erg, plot_cell_voltages, plot_ops, plot_intensity_response,
    plot_erg_gl, plot_cell_voltages_gl, plot_phototransduction_breakdown,
    # Validation
    compute_validation_metrics,
    # Utilities
    mean_nt, weighted_mean, synaptic_current,
    hc_feedback, dopamine_gain_factor

end # module
