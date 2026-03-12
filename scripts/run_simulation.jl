#!/usr/bin/env julia
#
# Command-line interface for RetinalTwin ERG simulation.
# Designed to be called from Python (or any other language) via subprocess.
#
# Usage:
#   julia --project run_simulation.jl \
#       --structure path/to/structure.json \
#       --params path/to/retinal_params.csv \
#       --stimulus path/to/stimulus_table.csv \
#       --depth path/to/erg_depth_map.csv \
#       --tspan 0.0,6.0 \
#       --dt 0.01 \
#       --outdir results/
#
# All flags are optional — defaults point to examples/inputs/default/.

using ArgParse
using RetinalTwin
using DifferentialEquations
using CSV, DataFrames

function parse_commandline()
    s = ArgParseSettings(
        description = "Run RetinalTwin ERG simulation from input files.",
        prog = "run_simulation.jl",
    )

    @add_arg_table! s begin
        "--structure"
            help = "Path to column structure JSON file"
            arg_type = String
            default = ""
        "--params"
            help = "Path to retinal parameters CSV file"
            arg_type = String
            default = ""
        "--stimulus"
            help = "Path to stimulus table CSV file"
            arg_type = String
            default = ""
        "--depth"
            help = "Path to ERG depth map CSV file"
            arg_type = String
            default = ""
        "--tspan"
            help = "Simulation time span as start,end (seconds)"
            arg_type = String
            default = "0.0,6.0"
        "--dt"
            help = "Output time step (seconds)"
            arg_type = Float64
            default = 0.01
        "--response-window"
            help = "Response window as start,end (seconds) for peak detection"
            arg_type = String
            default = "0.5,1.5"
        "--outdir"
            help = "Output directory for results"
            arg_type = String
            default = "."
        "--verbose"
            help = "Print progress information"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # --- Resolve input paths (default to examples/inputs/default/) ---
    default_dir = joinpath(dirname(@__DIR__), "examples", "inputs", "default")

    structure_fn = isempty(args["structure"]) ?
        joinpath(default_dir, "default_column.json") : args["structure"]
    params_fn = isempty(args["params"]) ?
        joinpath(default_dir, "retinal_params.csv") : args["params"]
    stimulus_fn = isempty(args["stimulus"]) ?
        joinpath(default_dir, "stimulus_table.csv") : args["stimulus"]
    depth_fn = isempty(args["depth"]) ?
        joinpath(default_dir, "erg_depth_map.csv") : args["depth"]

    # --- Parse tspan and response window ---
    tspan_parts = split(args["tspan"], ",")
    tspan = (parse(Float64, tspan_parts[1]), parse(Float64, tspan_parts[2]))

    rw_parts = split(args["response-window"], ",")
    response_window = (parse(Float64, rw_parts[1]), parse(Float64, rw_parts[2]))

    dt = args["dt"]
    outdir = args["outdir"]
    verbose = args["verbose"]

    # --- Validate inputs ---
    for (label, path) in [
        ("structure", structure_fn),
        ("params", params_fn),
        ("stimulus", stimulus_fn),
        ("depth", depth_fn),
    ]
        if !isfile(path)
            error("$label file not found: $path")
        end
    end

    if verbose
        println("Structure: $structure_fn")
        println("Params:    $params_fn")
        println("Stimulus:  $stimulus_fn")
        println("Depth:     $depth_fn")
        println("tspan:     $tspan")
        println("dt:        $dt")
        println("outdir:    $outdir")
    end

    # --- Run simulation ---
    t, erg_traces, solutions, peak_amps = simulate_erg(;
        structure = structure_fn,
        params = params_fn,
        stimulus_table = stimulus_fn,
        depth_csv = depth_fn,
        tspan = tspan,
        dt = dt,
        response_window = response_window,
        verbose = verbose,
    )

    # --- Write output ---
    mkpath(outdir)

    # 1) ERG traces CSV: columns = time, sweep_1, sweep_2, ...
    t_vec = collect(t)
    trace_df = DataFrame(:time => t_vec)
    for (i, tr) in enumerate(erg_traces)
        trace_df[!, Symbol("sweep_$i")] = tr
    end
    trace_path = joinpath(outdir, "erg_traces.csv")
    CSV.write(trace_path, trace_df)

    # 2) Peak amplitudes CSV
    stimulus_table = load_stimulus_table(stimulus_fn)
    amp_df = DataFrame(
        sweep = 1:length(peak_amps),
        intensity = [s.intensity for s in stimulus_table],
        peak_amplitude = peak_amps,
    )
    amp_path = joinpath(outdir, "peak_amplitudes.csv")
    CSV.write(amp_path, amp_df)

    if verbose
        println("\nSimulation complete.")
        println("  ERG traces:      $trace_path")
        println("  Peak amplitudes: $amp_path")
        println("  Sweeps:          $(length(erg_traces))")
        println("  Time points:     $(length(t_vec))")
    end

    # Print paths to stdout for easy parsing by calling process
    println(trace_path)
    println(amp_path)
end

main()
