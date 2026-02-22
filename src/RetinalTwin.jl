module RetinalTwin

using LinearAlgebra
using JSON
using Statistics

# --- Core types ---
# --- Default parameters ---
include("parameters/parameter_extraction.jl")
include("auxiliary_functions.jl")
include("current_equations.jl")
export
    # Parameter loading
    load_params_from_csv,
    default_rod_params,
    default_hc_params, default_on_bc_params, default_off_bc_params,
    default_a2_params, default_a2_amacrine_params, default_gaba_params, default_da_params, default_gc_params,
    default_muller_params, default_rpe_params,
    dict_to_namedtuple, namedtuple_to_dict,
    load_all_params,
    current_equations_for

# --- Cell update functions ---
include("cells/photoreceptor.jl")
export photoreceptor_state, photoreceptor_model!, photoreceptor_K_efflux, I_photoreceptor
include("cells/horizontal.jl")
export horizontal_state, horizontal_model!, I_horizontal
include("cells/on_bipolar.jl")
export on_bipolar_state, on_bipolar_model!, on_bipolar_K_efflux, I_on_bipolar
include("cells/off_bipolar.jl")
export off_bipolar_state, off_bipolar_model!, off_bipolar_K_efflux, I_off_bipolar
include("cells/a2_amacrine.jl")
export a2_amacrine_state, a2_amacrine_model!, a2_amacrine_K_efflux, I_a2_amacrine
include("cells/gaba_amacrine.jl")
export gaba_amacrine_state, gaba_amacrine_model!, I_gaba_amacrine
include("cells/da_amacrine.jl")
export da_amacrine_state, da_amacrine_model!, I_da_amacrine
include("cells/ganglion.jl")
export ganglion_state, ganglion_model!, ganglion_K_efflux, I_ganglion
include("cells/muller.jl")
export muller_state, muller_model!, I_muller
include("cells/rpe.jl")
export rpe_state, rpe_model!, I_rpe

#--- Circuit Mapping
include("circuit/mapping.jl")   
export CellRef, RetinalColumnModel, global_idx, display_global_idxs
export cell_range, uview, duview, get_out, connect!, square_grid_coords, build_column
export save_mapping, load_mapping


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
export make_uniform_flash_stimulus, exponential_spot_stimulus, make_exponential_spot_stimulus

# --- Fitting ---
include("fitting/fitting.jl")
include("fitting/gradients.jl")
export hill_ir, fit_hill_ir, calculate_ir_gradient, run_gradient_calculation

# # --- ERG ---
include("circuit/field_potential.jl")
export default_depth_csv_path, load_erg_depth_map, load_depth_map, load_depth_scales, compute_field_potential

# --- Visualization ---
include("visualization/plots.jl")
include("visualization/gl_plots.jl")

# --- Validation ---
include("validation/targets.jl")

end # module
