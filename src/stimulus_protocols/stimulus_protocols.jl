function uniform_flash(t; stim_start=0.0, stim_end=1.0, photon_flux=1.0, hold_flux=0.0)
    return (stim_start <= t <= stim_end) ? photon_flux : hold_flux
end

uniform_flash(t, x, y; kwargs...) = uniform_flash(t; kwargs...)

make_uniform_flash_stimulus(; kwargs...) = (t, x, y) -> uniform_flash(t, x, y; kwargs...)

function spatial_stimulus(t, x, y; xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, kwargs...)
    inside = (xmin <= x <= xmax) && (ymin <= y <= ymax)
    return inside ? uniform_flash(t; kwargs...) : 0.0
end

function exponential_spot_stimulus(
    t,
    x,
    y;
    center_x::Real=0.0,
    center_y::Real=0.0,
    decay_length::Real=1.0,
    stim_start::Real=0.0,
    stim_end::Real=1.0,
    photon_flux::Real=1.0,
    hold_flux::Real=0.0,
)
    decay_length > 0 || error("decay_length must be > 0, got $decay_length")
    base = uniform_flash(t; stim_start=stim_start, stim_end=stim_end, photon_flux=photon_flux, hold_flux=hold_flux)
    r = hypot(Float64(x) - Float64(center_x), Float64(y) - Float64(center_y))
    return base * exp(-r / Float64(decay_length))
end

make_exponential_spot_stimulus(; kwargs...) = (t, x, y) -> exponential_spot_stimulus(t, x, y; kwargs...)
