"""
    dark_adapt(model, u0, params; time=2000.0, abstol=1e-8, reltol=1e-6)

Run the model with zero stimulus to reach the dark-adapted steady state.

Uses `SteadyStateProblem` + `DynamicSS` to stop as soon as the derivative norm
falls below tolerance — much faster than simulating a fixed `time` duration.
`time` sets the maximum simulation time as a fallback if convergence is not
reached early.
"""
function dark_adapt(model, u0, params; time = 2000.0, abstol=1e-8, reltol=1e-6, verbose=false)
    stim_dark = make_uniform_flash_stimulus(photon_flux=0.0)
    ode_prob = DifferentialEquations.ODEProblem(model, u0, (0.0, time), (params, stim_dark))
    ss_prob = DifferentialEquations.SteadyStateProblem(ode_prob)
    if verbose
        println("Running dark adaptation to find steady state...")
        println("This may take a moment, but will stop as soon as convergence is reached.")
        @time sol = DifferentialEquations.solve(
            ss_prob,
            DifferentialEquations.DynamicSS(DifferentialEquations.Rodas5());
            abstol=abstol, reltol=reltol,
        )
    else
        sol = DifferentialEquations.solve(
            ss_prob,
            DifferentialEquations.DynamicSS(DifferentialEquations.Rodas5());
            abstol=abstol, reltol=reltol,
        )
    end
    return sol.u
end


"""
    simulate_erg(model, u0_dark, params; stim_duration=10.0, stim_intensity=1000.0,
                 tspan=(0.0, 6.0), dt=0.01, response_window=(0.5, 1.5), stim_start=0.0, verbose=false)

Simulate a single ERG trace for one stimulus flash.

# Arguments
- `model`: RetinalColumnModel
- `u0_dark`: dark-adapted initial state
- `params`: model parameters NamedTuple
- `stim_duration`: flash duration in ms (default 10.0)
- `stim_intensity`: photon flux intensity (default 1000.0)
- `stim_start`: stimulus onset time in seconds (default 0.0)

# Returns
`(t_grid, erg_trace, solution, peak_amp)`
"""
function simulate_erg(model, u0_dark, params;
    stim_duration=10.0, stim_intensity=1000.0,
    tspan=(0.0, 6.0), dt=0.01,
    response_window=(0.5, 1.5), stim_start=0.0,
    verbose=false
)
    t_rng = tspan[1]:dt:tspan[2]

    stim = make_uniform_flash_stimulus(
        stim_start=stim_start,
        stim_end=stim_start + stim_duration / 10.0,
        photon_flux=stim_intensity,
    )

    prob = DifferentialEquations.ODEProblem(model, u0_dark, tspan, (params, stim))
    sol = DifferentialEquations.solve(
        prob, DifferentialEquations.Rodas5();
        tstops=[stim_start, stim_start + stim_duration / 1000.0],
        abstol=1e-8, reltol=1e-6,
    )

    t_erg, erg = compute_field_potential(model, params, sol; dt=dt)

    response_idx = findall(t -> response_window[1] <= t <= response_window[2], t_rng)
    peak_amp = -minimum(erg[response_idx])

    verbose && println(
        "Intensity $(round(stim_intensity, digits=3)) $(stim_duration)ms: retcode=$(sol.retcode), " *
        "a-wave amp=$(round(peak_amp, digits=4))"
    )

    return t_rng, erg, sol, peak_amp
end

"""
    simulate_erg(model, u0_dark, params, stim_model; tspan=(0.0, 6.0), dt=0.01, ...)

Simulate ERG traces for multiple stimulus intensities.

# Arguments
- `stim_model`: Vector of NamedTuples with fields `intensity` and `duration`
- Returns `(t_grid, traces, solutions, peak_amps)`

Failed ODE solves produce NaN-filled traces rather than throwing.
"""
function simulate_erg(model, u0_dark, params, stim_model;
    tspan=(0.0, 6.0), dt=0.01,
    response_window=(0.5, 1.5), stim_start=0.0,
    verbose=false
)
    t_rng = tspan[1]:dt:tspan[2]
    erg_traces = Vector{Vector{Float64}}(undef, length(stim_model))
    solutions = Vector{Union{DifferentialEquations.ODESolution, Nothing}}(undef, length(stim_model))
    peak_amps = fill(NaN, length(stim_model))
    verbose && println("Simulating ERG traces for $(length(stim_model)) stimuli...")
    for (i, s) in enumerate(stim_model)
        t_rng, erg_traces[i], solutions[i], peak_amps[i] = simulate_erg(
            model, u0_dark, params;
            stim_duration=s.duration,
            stim_intensity=s.intensity,
            tspan=tspan, dt=dt,
            response_window=response_window,
            stim_start=stim_start,
            verbose=verbose,
        )
    end

    return t_rng, erg_traces, solutions, peak_amps
end