# ============================================================
# retinal_column.jl — Retinal column model with direct coupling
# ============================================================

# ── State organization ──────────────────────────────────────

DEFAULT_INDEXES = (
    PC_ICS_ID_BEGIN = 1,
    PC_ICS_ID_END = 21,
    ONBC_ICS_ID_BEGIN = 22,
    ONBC_ICS_ID_END = 27,
    OFFBC_ICS_ID_BEGIN = 28,
    OFFBC_ICS_ID_END = 34,
    A2_ICS_ID_BEGIN = 35,
    A2_ICS_ID_END = 41,
    GC_ICS_ID_BEGIN = 42,
    GC_ICS_ID_END = 47,
    #Next we can define the neurotransmitter indices
    PC_GLU_INDEX = 21,
    ONBC_GLU_INDEX = 6,
    OFFBC_GLU_INDEX = 7,
    A2_GLY_INDEX = 7
)
# ── Default parameters ──────────────────────────────────────

"""
    default_retinal_params()

Load default parameters for photoreceptor and ON bipolar cells.

# Returns
NamedTuple with:
- `PHOTORECEPTOR_PARAMS`: Rod photoreceptor parameters
- `ON_BIPOLAR_PARAMS`: ON bipolar cell parameters
- `OFF_BIPOLAR_PARAMS`: OFF bipolar cell parameters
- `A2_AMACRINE_PARAMS`: A2 amacrine cell parameters
"""
function default_retinal_params()
    return (
        PHOTORECEPTOR_PARAMS = default_rod_params(),
        ON_BIPOLAR_PARAMS = default_on_bc_params(),
        OFF_BIPOLAR_PARAMS = default_off_bc_params(),
        A2_AMACRINE_PARAMS = default_a2_params(),
        GANGLION_PARAMS = default_gc_params()
    )
end

# ── Initial conditions ──────────────────────────────────────

"""
    retinal_column_initial_conditions(params)

Build initial conditions for photoreceptor + ON bipolar system.

# Arguments
- `params`: NamedTuple from `default_retinal_params()`

# Returns
- 25-element state vector [photoreceptor(21), on_bipolar(6), off_bipolar(7), a2(7)]
"""
function retinal_column_initial_conditions(params)
    # Get individual cell initial conditions
    ic_size = 0
    u0_photoreceptor = photoreceptor_state(params.PHOTORECEPTOR_PARAMS)
    println("Size of photoreceptor state vector: $(length(u0_photoreceptor))")
    ic_size += length(u0_photoreceptor)
    println("IC size after photoreceptor: $ic_size")

    u0_on_bipolar = onbc_state(params.ON_BIPOLAR_PARAMS)
    println("Size of on bipolar state vector: $(length(u0_on_bipolar))")
    ic_size += length(u0_on_bipolar)
    println("IC size after on bipolar: $ic_size")

    u0_off_bipolar = offbc_state(params.OFF_BIPOLAR_PARAMS)
    println("Size of off bipolar state vector: $(length(u0_off_bipolar))")
    ic_size += length(u0_off_bipolar)
    println("IC size after off bipolar: $ic_size")

    u0_a2 = a2_state(params.A2_AMACRINE_PARAMS)
    println("Size of a2 state vector: $(length(u0_a2))")
    ic_size += length(u0_a2)
    println("IC size after a2: $ic_size")

    u0_gc = ganglion_dark_state(params.GANGLION_PARAMS)
    println("Size of ganglion state vector: $(length(u0_gc))")
    ic_size += length(u0_gc)
    println("IC size after ganglion: $ic_size")

    println("Total size of initial condition vector: $ic_size")
    # Concatenate into single state vector
    return vcat(u0_photoreceptor, u0_on_bipolar, u0_off_bipolar, u0_a2, u0_gc)
end

# ── Auxillary Functions ─────────────────────────────────────
#The only auxillary function we need to worry about is the gap junction coupling function

function gap_junction_coupling(dV1, V1, dV2, V2, cm1, cm2, g_gap)
    I_gap1 = g_gap * (V1 - V2)
    dV1 = -I_gap1 / cm1
    dV2 = I_gap1 / cm2
    return nothing
end


# ── Main model function ─────────────────────────────────────

"""
    retinal_column_model!(du, u, p, t)

ODE right-hand side for photoreceptor → ON bipolar cell system.

# Arguments
- `du`: derivative vector (25 elements)
- `u`: state vector (25 elements)
- `p`: tuple `(params, stim_params)` where:
  - `params`: NamedTuple from `default_retinal_params()`
  - `stim_params`: NamedTuple with stimulus information (stim_start, stim_end, photon_flux, v_hold)
- `t`: time (ms)

# State vector organization
```
u[1:21]   → Photoreceptor states [R, T, P, G, HC1-5, mKv, hKv, mCa, mKCa, Ca_s, Ca_f, CaB_ls, CaB_hs, CaB_lf, CaB_hf, V, Glu]
u[22:27]  → ON bipolar states [V, n, h, c, S, Glu_release]
u[28:34]  → OFF bipolar states [V, n, h, c, A, D, Glu_release]
u[35:41]  → A2 amacrine states [V, n, h, c, A, D, Gly_Release]
```

# Notes
- Photoreceptor computes its own glutamate release (u[21])
- Glutamate is passed directly to ON bipolar cell
- ON bipolar inverts signal via mGluR6 cascade
- A2 amacrine cell receives glutamate from ON bipolar cell and releases glycine
"""
function retinal_column_model!(du, u, p, t)
    params, stim_func = p

    idxs = DEFAULT_INDEXES
    # === Extract state segments using views (efficient, no copying) ===
    u_photoreceptor = @view u[idxs.PC_ICS_ID_BEGIN:idxs.PC_ICS_ID_END]
    u_on_bipolar = @view u[idxs.ONBC_ICS_ID_BEGIN:idxs.ONBC_ICS_ID_END]
    u_off_bipolar = @view u[idxs.OFFBC_ICS_ID_BEGIN:idxs.OFFBC_ICS_ID_END]
    u_a2 = @view u[idxs.A2_ICS_ID_BEGIN:idxs.A2_ICS_ID_END]
    u_gc = @view u[idxs.GC_ICS_ID_BEGIN:idxs.GC_ICS_ID_END]

    du_photoreceptor = @view du[idxs.PC_ICS_ID_BEGIN:idxs.PC_ICS_ID_END]
    du_on_bipolar = @view du[idxs.ONBC_ICS_ID_BEGIN:idxs.ONBC_ICS_ID_END]
    du_off_bipolar = @view du[idxs.OFFBC_ICS_ID_BEGIN:idxs.OFFBC_ICS_ID_END]
    du_a2 = @view du[idxs.A2_ICS_ID_BEGIN:idxs.A2_ICS_ID_END]
    du_gc = @view du[idxs.GC_ICS_ID_BEGIN:idxs.GC_ICS_ID_END]
    # === Neurotransmitter coupling ===

    # Get glutamate release from photoreceptor state
    glu_photo_release = u_photoreceptor[idxs.PC_GLU_INDEX]

    # === Call individual cell models ===

    # Photoreceptor
    photoreceptor_model!(du_photoreceptor, u_photoreceptor, (params.PHOTORECEPTOR_PARAMS, stim_func), t)

    # ON bipolar (receives glutamate from photoreceptor)
    on_bipolar_model!(du_on_bipolar, u_on_bipolar, (params.ON_BIPOLAR_PARAMS, glu_photo_release), t)
    glu_on_bipolar_release = u_on_bipolar[idxs.ONBC_GLU_INDEX]

    # OFF bipolar (receives glutamate from photoreceptor)
    off_bipolar_model!(du_off_bipolar, u_off_bipolar, (params.OFF_BIPOLAR_PARAMS, glu_photo_release), t)
    glu_off_bipolar_release = u_off_bipolar[idxs.OFFBC_GLU_INDEX]

    # A2 amacrine cell (receives glutamate from ON bipolar cell)
    a2_model!(du_a2, u_a2, (params.A2_AMACRINE_PARAMS, glu_on_bipolar_release), t)
    gly_a2_release = u_a2[idxs.A2_GLY_INDEX]

    # Ganglion cell (receives excitatory bipolar and inhibitory A2 amacrine drive)
    glu_gc_exc = glu_on_bipolar_release + glu_off_bipolar_release
    ganglion_model!(du_gc, u_gc, (params.GANGLION_PARAMS, glu_gc_exc, gly_a2_release), t)

    #Electrical coupling between cells
    gap_junction_coupling(du_a2[1], u_a2[1], du_on_bipolar[1], u_on_bipolar[1], params.A2_AMACRINE_PARAMS.C_m, params.ON_BIPOLAR_PARAMS.C_m, params.A2_AMACRINE_PARAMS.g_gap)

    return nothing
end
