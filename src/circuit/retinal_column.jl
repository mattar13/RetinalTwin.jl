# ============================================================
# retinal_column.jl — Retinal column model with direct coupling
# ============================================================

# ── State organization ──────────────────────────────────────
# For single photoreceptor + single ON bipolar cell:
# u = [photoreceptor_states(21), on_bipolar_states(4)]
# Total: 25 state variables

const PHOTORECEPTOR_OFFSET = 0
const PHOTORECEPTOR_SIZE = 21

const ON_BIPOLAR_OFFSET = 21
const ON_BIPOLAR_SIZE = 4

# ── Default parameters ──────────────────────────────────────

"""
    default_retinal_params()

Load default parameters for photoreceptor and ON bipolar cells.

# Returns
NamedTuple with:
- `PHOTORECEPTOR_PARAMS`: Rod photoreceptor parameters
- `ON_BIPOLAR_PARAMS`: ON bipolar cell parameters
"""
function default_retinal_params()
    return (
        PHOTORECEPTOR_PARAMS = default_rod_params_csv(),
        ON_BIPOLAR_PARAMS = default_on_bc_params_csv()
    )
end

# ── Initial conditions ──────────────────────────────────────

"""
    retinal_column_initial_conditions(params)

Build initial conditions for photoreceptor + ON bipolar system.

# Arguments
- `params`: NamedTuple from `default_retinal_params()`

# Returns
- 25-element state vector [photoreceptor(21), on_bipolar(4)]
"""
function retinal_column_initial_conditions(params)
    # Get individual cell initial conditions
    u0_photoreceptor = rod_dark_state(params.PHOTORECEPTOR_PARAMS)
    u0_on_bipolar = on_bipolar_dark_state(params.ON_BIPOLAR_PARAMS)

    # Concatenate into single state vector
    return vcat(u0_photoreceptor, u0_on_bipolar)
end

# ── Auxiliary functions ─────────────────────────────────────

"""
    compute_glutamate_release(V, V_half, V_slope, alpha)

Compute glutamate release from photoreceptor voltage using sigmoid.
Higher (less negative) voltage → more glutamate release.
"""
function compute_glutamate_release(V::Real, V_half::Real, V_slope::Real, alpha::Real)
    return alpha / (1.0 + exp(-(V - V_half) / V_slope))
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
u[22:25]  → ON bipolar states [V, w, S_mGluR6, Glu_release]
```

# Notes
- Photoreceptor computes its own glutamate release (u[21])
- Glutamate is passed directly to ON bipolar cell
- ON bipolar inverts signal via mGluR6 cascade
"""
function retinal_column_model!(du, u, p, t)
    params, stim_params = p

    # === Extract state segments using views (efficient, no copying) ===
    u_photoreceptor = @view u[1:21]
    u_on_bipolar = @view u[22:25]

    du_photoreceptor = @view du[1:21]
    du_on_bipolar = @view du[22:25]

    # === Neurotransmitter coupling ===

    # Get glutamate release from photoreceptor state
    glu_release = u_photoreceptor[ROD_GLU_INDEX]

    # === Call individual cell models ===

    # Photoreceptor
    rod_model!(du_photoreceptor, u_photoreceptor,
               (params.PHOTORECEPTOR_PARAMS, stim_params), t)

    # ON bipolar (receives glutamate from photoreceptor)
    on_bipolar_model!(du_on_bipolar, u_on_bipolar,
                      (params.ON_BIPOLAR_PARAMS, glu_release), t)

    return nothing
end
