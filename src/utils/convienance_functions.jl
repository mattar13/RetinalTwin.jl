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


# ── Internal single-stimulus simulation (used by multi-stimulus and fitting) ──

"""
    _simulate_single_erg(model, u0_dark, params; stim_duration, stim_intensity,
                         stim_start, tspan, dt, response_window, depth_csv, verbose)

Internal: simulate a single ERG trace for one stimulus flash.
Returns `(t_grid, erg_trace, solution, peak_amp)`.
"""
function _simulate_single_erg(model, u0_dark, params;
    stim_duration=10.0, stim_intensity=1000.0,
    stim_start=0.0,
    tspan=(0.0, 6.0), dt=0.01,
    response_window=(0.5, 1.5),
    depth_csv::AbstractString=default_depth_csv_path(),
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

    t_erg, erg = compute_field_potential(model, params, sol; depth_csv=depth_csv, dt=dt)

    response_idx = findall(t -> response_window[1] <= t <= response_window[2], t_rng)
    peak_amp = -minimum(erg[response_idx])

    verbose && println(
        "Intensity $(round(stim_intensity, digits=3)) $(stim_duration)ms: retcode=$(sol.retcode), " *
        "a-wave amp=$(round(peak_amp, digits=4))"
    )

    return t_rng, erg, sol, peak_amp
end


# ── Backwards-compatible positional methods (used by fitting code) ────────────

"""
    simulate_erg(model, u0_dark, params; stim_duration, stim_intensity, ...)

Simulate a single ERG trace for one stimulus flash (positional model/params form).
Returns `(t_grid, erg_trace, solution, peak_amp)`.
"""
function simulate_erg(model::RetinalColumnModel, u0_dark::AbstractVector, params::NamedTuple;
    stim_duration=10.0, stim_intensity=1000.0,
    tspan=(0.0, 6.0), dt=0.01,
    response_window=(0.5, 1.5), stim_start=0.0,
    depth_csv::AbstractString=default_depth_csv_path(),
    verbose=false
)
    return _simulate_single_erg(model, u0_dark, params;
        stim_duration=stim_duration, stim_intensity=stim_intensity,
        stim_start=stim_start, tspan=tspan, dt=dt,
        response_window=response_window, depth_csv=depth_csv, verbose=verbose,
    )
end

"""
    simulate_erg(model, u0_dark, params, stim_model; tspan, dt, ...)

Simulate ERG traces for multiple stimulus intensities (positional model/params form).
`stim_model` is a Vector of NamedTuples with fields `intensity` and `duration`.
Returns `(t_grid, traces, solutions, peak_amps)`.
"""
function simulate_erg(model::RetinalColumnModel, u0_dark::AbstractVector, params::NamedTuple, stim_model::AbstractVector;
    tspan=(0.0, 6.0), dt=0.01,
    response_window=(0.5, 1.5), stim_start=0.0,
    depth_csv::AbstractString=default_depth_csv_path(),
    verbose=false
)
    t_rng = tspan[1]:dt:tspan[2]
    erg_traces = Vector{Vector{Float64}}(undef, length(stim_model))
    solutions = Vector{Union{DifferentialEquations.ODESolution, Nothing}}(undef, length(stim_model))
    peak_amps = fill(NaN, length(stim_model))
    verbose && println("Simulating ERG traces for $(length(stim_model)) stimuli...")
    for (i, s) in enumerate(stim_model)
        s_start = hasproperty(s, :time_start) ? s.time_start : stim_start
        t_rng, erg_traces[i], solutions[i], peak_amps[i] = _simulate_single_erg(
            model, u0_dark, params;
            stim_duration=s.duration,
            stim_intensity=s.intensity,
            stim_start=s_start,
            tspan=tspan, dt=dt,
            response_window=response_window,
            depth_csv=depth_csv,
            verbose=verbose,
        )
    end

    return t_rng, erg_traces, solutions, peak_amps
end


# ── Top-level keyword-based entry point ───────────────────────────────────────

"""
    simulate_erg(; structure, params, stimulus_table, depth_csv, tspan, dt, response_window, verbose)

High-level ERG simulation that handles file loading and dark adaptation automatically.

Each of `structure`, `params`, and `stimulus_table` can be either:
- A `String` file path → loaded from disk via `load_mapping`, `load_all_params`,
  or `load_stimulus_table`
- An already-loaded object → used directly:
  - `structure`: `Tuple{RetinalColumnModel, Vector}` from `load_mapping`
  - `params`: `NamedTuple` (immutable) or `Dict` (mutable, converted via `dict_to_namedtuple`)
  - `stimulus_table`: `Vector` of NamedTuples with `intensity`, `duration` (and optionally `time_start`)

The model is automatically dark-adapted before simulation.

Returns `(t_grid, erg_traces, solutions, peak_amps)`.
"""
function simulate_erg(;
    structure::Union{AbstractString, Tuple},
    params::Union{AbstractString, NamedTuple, AbstractDict},
    stimulus_table::Union{AbstractString, AbstractVector},
    depth_csv::AbstractString = default_depth_csv_path(),
    tspan = (0.0, 6.0),
    dt = 0.01,
    response_window = (0.5, 1.5),
    dark_adapt_time = 2000.0,
    verbose = false,
)
    # Load or unpack structure
    if structure isa AbstractString
        model, u0 = load_mapping(structure)
    else
        model, u0 = structure
    end

    # Load or convert params
    if params isa AbstractString
        p = load_all_params(csv_path = params)
    elseif params isa AbstractDict
        p = dict_to_namedtuple(params)
    else
        p = params
    end

    # Load or use stimulus table
    if stimulus_table isa AbstractString
        stim_model = load_stimulus_table(stimulus_table)
    else
        stim_model = stimulus_table
    end

    # Dark adapt
    verbose && println("Dark adapting...")
    u0_dark = dark_adapt(model, u0, p; time=dark_adapt_time)

    # Simulate all stimuli
    return simulate_erg(model, u0_dark, p, stim_model;
        tspan=tspan, dt=dt, response_window=response_window,
        depth_csv=depth_csv, verbose=verbose,
    );
end
