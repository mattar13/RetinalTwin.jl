using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics
using Dates

println("=" ^ 70)
println("A-wave parameter fitting (Optim.jl — CMA-ES + LBFGS)")
println("=" ^ 70)

# -----------------------------------------------------------------------------
#%% 1) Input file paths (all from examples/inputs/)
# -----------------------------------------------------------------------------
input_dir    = joinpath(@__DIR__, "inputs", "default")
structure_fn = joinpath(input_dir, "photoreceptor_column.json")
params_fn    = joinpath(input_dir, "retinal_params.csv")
stimulus_fn  = joinpath(input_dir, "stimulus_table.csv")
depth_fn     = joinpath(input_dir, "erg_depth_map.csv")

# -----------------------------------------------------------------------------
#%% 2) Load inputs
# -----------------------------------------------------------------------------
model, u0 = load_mapping(structure_fn)
stimulus_table = load_stimulus_table(stimulus_fn)

# Load params as mutable dict so we can apply blockers
params_dict = load_all_params(csv_path = params_fn, editable = true)
params_dict[:ONBC][:g_TRPM1]      = 0.0
params_dict[:OFFBC][:g_iGluR]     = 0.0
params_dict[:MULLER][:g_Kir_end]   = 0.0
params_dict[:MULLER][:g_Kir_stalk] = 0.0
params = dict_to_namedtuple(params_dict)

println("\nCells in model: ", ordered_cells(model))
println("Total states:   ", length(u0))
println("Stimuli:        ", length(stimulus_table), " sweeps")

# -----------------------------------------------------------------------------
#%% 3) Load real ERG data for comparison
# -----------------------------------------------------------------------------
using ElectroPhysiology
import ElectroPhysiology: save_stimulus_csv

response_window = (0.5, 1.5)
erg_dir = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods"
erg_files = parseABF(erg_dir)
real_dataset = openERGData(erg_files, t_post = 9.74)

# Save stimulus table from real data (keeps inputs/ in sync)
save_stimulus_csv(stimulus_fn, real_dataset)
stimulus_table = load_stimulus_table(stimulus_fn)

n_sweeps = eachtrial(real_dataset) |> length
real_traces = Vector{Vector{Float64}}(undef, n_sweeps)
real_t_vec  = Vector{Vector{Float64}}(undef, n_sweeps)
real_amp    = fill(NaN, n_sweeps)

for (i, sweep) in enumerate(eachtrial(real_dataset))
    t = sweep.t
    y = sweep.data_array[1,:,1] * 1000.0
    real_traces[i] = y
    real_t_vec[i]  = t
    first_peak_idx = findfirst(t .>= response_window[1])
    last_peak_idx  = findfirst(t .>= response_window[2])
    real_amp[i] = -minimum(y[first_peak_idx:last_peak_idx])
end

# -----------------------------------------------------------------------------
#%% 4) Run fitting
# -----------------------------------------------------------------------------
println("\n" * "=" ^ 70)
println("Starting a-wave parameter fitting...")
println("=" ^ 70)

tspan = (0.0, real_dataset.t[end])
dt = 0.01
checkpoint_base = joinpath(@__DIR__, "checkpoints")

result = fit_erg(
    model, u0, params;
    cell_types = [:PHOTO],
    stimuli = stimulus_table,
    real_t = real_t_vec,
    real_traces = real_traces,
    time_window = response_window,
    tspan = tspan,
    dt = dt,
    optimizer = :cma_es,
    phase1_iterations = 500,
    lbfgs_iterations = 100,
    run_lbfgs = true,
    checkpoint_every = 100,
    checkpoint_dir = checkpoint_base,
    checkpoint_files = Dict(
        structure_fn => "structure.json",
        stimulus_fn  => "stimulus_table.csv",
        depth_fn     => "erg_depth_map.csv",
    ),
    verbose = true,
)

# -----------------------------------------------------------------------------
#%% 5) Save final checkpoint — complete snapshot of all inputs + fitted outputs
# -----------------------------------------------------------------------------
final_dir = joinpath(checkpoint_base, "final_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))")
mkpath(final_dir)

# Save fitted params CSV
save_fitted_params_csv(result, joinpath(final_dir, "retinal_params.csv");
    template_csv = params_fn)

# Save uncertainty datasheet
save_fit_datasheet(result, joinpath(final_dir, "awave_fit_params.csv"))

# Copy structure, stimulus, and depth files into checkpoint
for (src, dst_name) in [
    (structure_fn, "structure.json"),
    (stimulus_fn,  "stimulus_table.csv"),
    (depth_fn,     "erg_depth_map.csv"),
]
    isfile(src) && cp(src, joinpath(final_dir, dst_name); force=true)
end

# Diagnostics plots
plots = plot_fit_diagnostics(
    result, real_t_vec, real_traces, stimulus_table;
    time_window = response_window,
    outdir = final_dir,
)

# -----------------------------------------------------------------------------
#%% 6) Summary
# -----------------------------------------------------------------------------
println("\n" * "=" ^ 70)
println("Fitting complete!")
println("=" ^ 70)
println("\nConvergence:")
println("  Optimizer:         $(result.convergence.optimizer)")
println("  Initial loss:      $(result.convergence.initial_loss)")
println("  Final loss:        $(result.convergence.final_loss)")
println("  Phase 1 converged: $(result.convergence.phase1_converged)")
println("  LBFGS converged:   $(result.convergence.lbfgs_converged)")
println("\nOutputs saved to: $final_dir")
println("  retinal_params.csv   (fitted parameters)")
println("  structure.json        (column architecture)")
println("  stimulus_table.csv    (stimulus protocol)")
println("  erg_depth_map.csv     (depth weighting)")
println("  awave_fit_params.csv  (uncertainty datasheet)")
println("  fit_traces.png")
println("  fit_residuals.png")
println("  fit_parameters.png")
if !isempty(plots.correlations)
    println("  fit_correlations.png")
end
