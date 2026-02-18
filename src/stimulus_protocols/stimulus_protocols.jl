function uniform_flash(t; stim_start=0.0, stim_end=1.0, photon_flux=1.0, hold_flux=0.0)
    return (stim_start <= t <= stim_end) ? photon_flux : hold_flux
end

uniform_flash(t, x, y; kwargs...) = uniform_flash(t; kwargs...)

function spatial_stimulus(t, x, y; xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, kwargs...)
    inside = (xmin <= x <= xmax) && (ymin <= y <= ymax)
    return inside ? uniform_flash(t; kwargs...) : 0.0
end
