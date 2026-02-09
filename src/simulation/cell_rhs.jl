# ============================================================
# cell_rhs.jl — Single-cell RHS wrappers for modular simulation
# ============================================================

"""
    rod_cell_rhs!(du, u, p, t)

Single rod photoreceptor RHS.
`p = (params::RodPhotoreceptorParams, Phi::Real, I_feedback::Real)`.
"""
function rod_cell_rhs!(du, u, p, t)
    params, Phi, I_feedback = p
    update_rod_photoreceptor!(du, u, params, Phi, I_feedback)
    return nothing
end

"""
    cone_cell_rhs!(du, u, p, t)

Single cone photoreceptor RHS.
`p = (params::PhototransductionParams, Phi::Real, I_feedback::Real)`.
"""
function cone_cell_rhs!(du, u, p, t)
    params, Phi, I_feedback = p
    update_photoreceptor!(du, u, params, Phi, I_feedback)
    return nothing
end

"""
    horizontal_cell_rhs!(du, u, p, t)

Single horizontal cell RHS.
`p = (params::MLParams, I_exc::Real, I_gap::Real, glu_mean::Real)`.
"""
function horizontal_cell_rhs!(du, u, p, t)
    params, I_exc, I_gap, glu_mean = p
    update_horizontal!(du, u, params, I_exc, I_gap)
    du[3] = (glu_mean - u[3]) / 3.0
    return nothing
end

"""
    on_bipolar_cell_rhs!(du, u, p, t)

Single ON bipolar RHS.
`p = (params::MLParams, mg::mGluR6Params, glu_mean::Real, I_inh::Real, I_mod::Real)`.
"""
function on_bipolar_cell_rhs!(du, u, p, t)
    params, mg, glu_mean, I_inh, I_mod = p
    update_on_bipolar!(du, u, params, mg, glu_mean, I_inh, I_mod)
    return nothing
end

"""
    off_bipolar_cell_rhs!(du, u, p, t)

Single OFF bipolar RHS.
`p = (params::MLParams, glu_mean::Real, I_inh::Real, I_mod::Real)`.
"""
function off_bipolar_cell_rhs!(du, u, p, t)
    params, glu_mean, I_inh, I_mod = p
    update_off_bipolar!(du, u, params, glu_mean, I_inh, I_mod)
    return nothing
end

"""
    a2_cell_rhs!(du, u, p, t)

Single A2 amacrine RHS.
`p = (params::MLParams, I_exc::Real, I_inh::Real, I_mod::Real)`.
"""
function a2_cell_rhs!(du, u, p, t)
    params, I_exc, I_inh, I_mod = p
    update_a2!(du, u, params, I_exc, I_inh, I_mod)
    return nothing
end

"""
    gaba_cell_rhs!(du, u, p, t)

Single GABAergic amacrine RHS.
`p = (params::MLParams, I_exc::Real, I_inh::Real, I_mod::Real)`.
"""
function gaba_cell_rhs!(du, u, p, t)
    params, I_exc, I_inh, I_mod = p
    update_gaba_amacrine!(du, u, params, I_exc, I_inh, I_mod)
    return nothing
end

"""
    da_cell_rhs!(du, u, p, t)

Single dopaminergic amacrine RHS.
`p = (params::MLParams, I_exc::Real)`.
"""
function da_cell_rhs!(du, u, p, t)
    params, I_exc = p
    update_da_amacrine!(du, u, params, I_exc)
    return nothing
end

"""
    ganglion_cell_rhs!(du, u, p, t)

Single ganglion cell RHS.
`p = (params::MLParams, I_exc::Real, I_inh::Real)`.
"""
function ganglion_cell_rhs!(du, u, p, t)
    params, I_exc, I_inh = p
    update_ganglion!(du, u, params, I_exc, I_inh)
    return nothing
end

"""
    muller_cell_rhs!(du, u, p, t)

Single Müller glia RHS.
`p = (params::MullerParams, I_K_neural::Real, Glu_release_total::Real)`.
"""
function muller_cell_rhs!(du, u, p, t)
    params, I_K_neural, Glu_release_total = p
    update_muller!(du, u, params, I_K_neural, Glu_release_total)
    return nothing
end

"""
    rpe_cell_rhs!(du, u, p, t)

Single RPE RHS.
`p = (params::RPEParams, I_K_PR::Real)`.
"""
function rpe_cell_rhs!(du, u, p, t)
    params, I_K_PR = p
    update_rpe!(du, u, params, I_K_PR)
    return nothing
end
