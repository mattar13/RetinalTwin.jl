module RetinalTwin

using LinearAlgebra
using Statistics

# --- Core types ---
# --- Default parameters ---
include("parameters/parameter_extraction.jl")
export
    # Parameter loading
    default_rod_params,
    default_hc_params, default_on_bc_params, default_off_bc_params,
    default_a2_params, default_gaba_params, default_da_params, default_gc_params,
    default_muller_params, default_rpe_params,
    load_all_params

# --- Cell update functions ---
include("cells/photoreceptor.jl")
export rod_dark_state, photoreceptor_model!
include("cells/horizontal.jl")
export horizontal_dark_state, horizontal_model!
include("cells/on_bipolar.jl")
export on_bipolar_dark_state, on_bipolar_model!
include("cells/off_bipolar.jl")
export off_bipolar_dark_state, off_bipolar_model!
include("cells/a2_amacrine.jl")
export a2_dark_state, a2_model!
include("cells/gaba_amacrine.jl")
export gaba_amacrine_dark_state, gaba_amacrine_model!
include("cells/da_amacrine.jl")
export da_amacrine_dark_state, da_amacrine_model!
include("cells/ganglion.jl")
export ganglion_dark_state, ganglion_model!
include("cells/muller.jl")
export muller_dark_state, muller_model!
include("cells/rpe.jl")
export rpe_dark_state, rpe_model!

# --- Circuit wiring ---
include("circuit/retinal_column.jl")
export
    # Retinal column state organization
    DEFAULT_INDEXES,
    # Retinal column (modular approach)
    default_retinal_params,
    retinal_column_initial_conditions,
    retinal_column_model!

# --- Stimulus protocols ---
include("stimulus_protocols/single_flash.jl")
export single_flash

# # --- ERG ---
# include("erg/field_potential.jl")

# --- Visualization ---
include("visualization/plots.jl")
include("visualization/gl_plots.jl")

# --- Validation ---
include("validation/targets.jl")

end # module
