module RetinalTwin

using LinearAlgebra
using Statistics

# --- Core types ---
include("types.jl")

# --- Default parameters ---
include("parameters/parameter_extraction.jl")
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
include("simulation/cell_rhs.jl")
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
    MLParams, PhototransductionParams, RodPhotoreceptorParams, NTReleaseParams,
    SynapseParams, mGluR6Params, MullerParams, RPEParams,
    ERGWeights, StimulusProtocol, PopulationSizes,
    RetinalColumn, StateIndex, ConnectionDef,
    # State layout constants
    ROD_STATE_VARS,
    ROD_R_INDEX, ROD_T_INDEX, ROD_P_INDEX, ROD_G_INDEX,
    ROD_HC1_INDEX, ROD_HC2_INDEX, ROD_HO1_INDEX, ROD_HO2_INDEX, ROD_HO3_INDEX,
    ROD_MKV_INDEX, ROD_HKV_INDEX, ROD_MCA_INDEX, ROD_MKCA_INDEX,
    ROD_CA_S_INDEX, ROD_CA_F_INDEX, ROD_CAB_LS_INDEX, ROD_CAB_HS_INDEX,
    ROD_CAB_LF_INDEX, ROD_CAB_HF_INDEX, ROD_V_INDEX,
    # Parameter factories
    build_retinal_column,
    default_rod_params,
    default_hc_params, default_on_bc_params, default_off_bc_params,
    default_a2_params, default_gaba_params, default_da_params, default_gc_params,
    default_muller_params, default_rpe_params, default_mglur6_params,
    default_erg_weights, default_connections,
    # Stimulus
    compute_stimulus, flash_stimulus,
    # Simulation
    simulate_flash, extract_voltages, extract_neurotransmitters,
    dark_adapted_state, retinal_column_rhs!, zipper_rhs!,
    # Photoreceptor models and initial conditions
    rod_dark_state, rod_model!,
    # Cell-level RHS functions
    rod_cell_rhs!, cone_cell_rhs!,
    horizontal_cell_rhs!, on_bipolar_cell_rhs!, off_bipolar_cell_rhs!,
    a2_cell_rhs!, gaba_cell_rhs!, da_cell_rhs!,
    muller_cell_rhs!, rpe_cell_rhs!, ganglion_cell_rhs!,
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
