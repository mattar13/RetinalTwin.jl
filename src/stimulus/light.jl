# ============================================================
# light.jl — Stimulus protocols
# ============================================================

"""
    compute_stimulus(stim::StimulusProtocol, t::Float64)

Return photon flux at time t. Box function for flash stimulus.
"""
function compute_stimulus(stim::StimulusProtocol, t::Real)
    if stim.t_on <= t <= stim.t_on + stim.t_dur
        return stim.I_0
    else
        return stim.background
    end
end

"""
    flash_stimulus(; intensity, t_on, duration, background)

Convenience constructor for a flash stimulus.
"""
function flash_stimulus(; intensity::Float64=1000.0, t_on::Float64=200.0,
                         duration::Float64=10.0, background::Float64=0.0)
    StimulusProtocol(I_0=intensity, t_on=t_on, t_dur=duration, background=background)
end
