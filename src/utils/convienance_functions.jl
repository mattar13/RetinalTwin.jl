"""
    _dark_adapt(model, u0, params; tspan_dark=(0.0, 2000.0), abstol=1e-6, reltol=1e-4)

Run the model with zero stimulus to reach dark-adapted steady state.
"""
function dark_adapt(model, u0, params; time = 2000.0, abstol=1e-6, reltol=1e-4)
    stim_dark = make_uniform_flash_stimulus(photon_flux=0.0)
    prob = DifferentialEquations.ODEProblem(model, u0, (0.0, time), (params, stim_dark))
    sol = DifferentialEquations.solve(
        prob, DifferentialEquations.Rodas5();
        save_everystep=false, save_start=false, save_end=true,
        abstol=abstol, reltol=reltol,
    )
    return sol.u[end]
end

"""
    simulate_erg(model, u0_dark, params, stimuli; tspan=(0.0, 6.0), dt=0.01)

Simulate ERG traces for multiple stimulus intensities.

# Arguments
- `stimuli`: Vector of NamedTuples with fields `intensity` and `duration_sec`
- Returns `(t_grid, traces)` where traces is a Vector{Vector{Float64}}

Failed ODE solves produce NaN-filled traces rather than throwing.
"""
function simulate_erg(model, u0_dark, params, stimuli; tspan=(0.0, 6.0), dt=0.01, response_window = (0.5, 1.5), stim_start = 0.0, verbose=true)
    t_rng = tspan[1]:dt:tspan[2]
    erg_traces = Vector{Vector{Float64}}(undef, length(stimuli))
    solutions = Vector{Union{DifferentialEquations.ODESolution, Nothing}}(undef, length(stimuli))
    peak_amps = fill(NaN, length(stimuli))
    verbose && println("Simulating ERG traces for $(length(stimuli)) stimuli...")
    for (i, s) in enumerate(stimuli)
        stim = make_uniform_flash_stimulus(
            stim_start=stim_start,
            stim_end=stim_start + s.duration/1000.0,
            photon_flux=s.intensity,
        )

        prob = DifferentialEquations.ODEProblem(model, u0_dark, tspan, (params, stim))
        sol = DifferentialEquations.solve(
            prob, DifferentialEquations.Rodas5();
            tstops=[stim_start, stim_start + s.duration/1000.0],
            abstol=1e-6, reltol=1e-4,
        )
        
        t_erg, erg = compute_field_potential(model, params, sol; dt=dt)
        
        #Store in data containers. 
        solutions[i] = sol
        erg_traces[i] = erg
        response_idx = findall(t -> response_window[1] <= t <= response_window[2], t_rng)
        peak_amps[i] = -minimum(erg[response_idx])


        verbose && println(
            "Intensity $(round(s.intensity, digits=3)) $(s.duration)ms: retcode=$(sol.retcode), " *
            "a-wave amp=$(round(peak_amps[i], digits=4))"
        )
    end

    return t_rng, erg_traces, solutions, peak_amps
end