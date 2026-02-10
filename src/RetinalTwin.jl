module RetinalTwin

using LinearAlgebra
using Statistics

# --- Core types ---
include("types.jl")

# --- Default parameters ---
include("parameters/parameter_extraction.jl")

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

# --- Circuit wiring ---
include("circuit/retinal_column.jl")

# --- ERG ---
include("erg/field_potential.jl")

# --- Visualization ---
include("visualization/plots.jl")
include("visualization/gl_plots.jl")

# --- Validation ---
include("validation/targets.jl")

# --- Public API ---
export
    # Photoreceptor state indices
    ROD_STATE_VARS,
    ROD_R_INDEX, ROD_T_INDEX, ROD_P_INDEX, ROD_G_INDEX,
    ROD_HC1_INDEX, ROD_HC2_INDEX, ROD_HO1_INDEX, ROD_HO2_INDEX, ROD_HO3_INDEX,
    ROD_MKV_INDEX, ROD_HKV_INDEX, ROD_MCA_INDEX, ROD_MKCA_INDEX,
    ROD_CA_S_INDEX, ROD_CA_F_INDEX, ROD_CAB_LS_INDEX, ROD_CAB_HS_INDEX,
    ROD_CAB_LF_INDEX, ROD_CAB_HF_INDEX, ROD_V_INDEX, ROD_GLU_INDEX,
    # Retinal column state organization
    PHOTORECEPTOR_OFFSET, PHOTORECEPTOR_SIZE,
    ON_BIPOLAR_OFFSET, ON_BIPOLAR_SIZE,
    # Parameter loading
    default_rod_params,
    default_hc_params, default_on_bc_params, default_off_bc_params,
    default_a2_params, default_gaba_params, default_da_params, default_gc_params,
    default_muller_params, default_rpe_params,
    load_all_params,
    # Retinal column (modular approach)
    default_retinal_params,
    retinal_column_initial_conditions,
    retinal_column_model!,
    # Individual cell models
    rod_dark_state, rod_model!,
    horizontal_dark_state, horizontal_model!,
    on_bipolar_dark_state, on_bipolar_model!,
    off_bipolar_dark_state, off_bipolar_model!,
    a2_amacrine_dark_state, a2_amacrine_model!,
    gaba_amacrine_dark_state, gaba_amacrine_model!,
    da_amacrine_dark_state, da_amacrine_model!,
    ganglion_dark_state, ganglion_model!,
    muller_dark_state, muller_model!,
    rpe_dark_state, rpe_model!

end # module
