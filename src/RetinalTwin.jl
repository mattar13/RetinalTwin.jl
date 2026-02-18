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
export photoreceptor_state, photoreceptor_model!, photoreceptor_K_efflux
include("cells/horizontal.jl")
export horizontal_state, horizontal_model!
include("cells/on_bipolar.jl")
export on_bipolar_state, on_bipolar_model!, on_bipolar_K_efflux
include("cells/off_bipolar.jl")
export off_bipolar_state, off_bipolar_model!, off_bipolar_K_efflux
include("cells/a2_amacrine.jl")
export a2_amacrine_state, a2_amacrine_model!, a2_amacrine_K_efflux
include("cells/gaba_amacrine.jl")
export gaba_amacrine_state, gaba_amacrine_model!
include("cells/da_amacrine.jl")
export da_amacrine_state, da_amacrine_model!
include("cells/ganglion.jl")
export ganglion_state, ganglion_model!, ganglion_K_efflux
include("cells/muller.jl")
export muller_state, muller_model!
include("cells/rpe.jl")
export rpe_state, rpe_model!

#--- Circuit Mapping
include("circuit/mapping.jl")   
export CellRef, RetinalColumnModel, global_idx, display_global_idxs
export cell_range, uview, duview, get_out, connect!, square_grid_coords, build_column


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
include("stimulus_protocols/stimulus_protocols.jl")
export single_flash, uniform_flash, spatial_stimulus

# # --- ERG ---
# include("erg/field_potential.jl")

# --- Visualization ---
include("visualization/plots.jl")
include("visualization/gl_plots.jl")

# --- Validation ---
include("validation/targets.jl")

end # module
