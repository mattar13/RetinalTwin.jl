using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics
using ElectroPhysiology
import ElectroPhysiology: parse_stimulus_name, calculate_photons, percent_to_photons_eq, nd_filter

println("=" ^ 70)
println("A-wave parameter fitting (Optim.jl — NelderMead + LBFGS)")
println("=" ^ 70)

#%%1) Load real BaCl+LAP4 ERG data via ElectroPhysiology.jl ────────────────
response_window = (0.5, 1.5)

erg_dir = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods"
erg_files = parseABF(erg_dir)
real_dataset = openERGData(erg_files, t_post = 6.0)

#Extract traces from ElectroPhysiology types into plain vectors ────────
n_sweeps = eachtrial(real_dataset) |> length
real_traces = Vector{Vector{Float64}}(undef, n_sweeps)
real_t = Vector{Vector{Float64}}(undef, n_sweeps)
real_amp = fill(NaN, n_sweeps)

for (i, sweep) in enumerate(eachtrial(real_dataset))
    t = sweep.t
    y = sweep.data_array[1,:,1] * 1000.0
    # println(sweep.HeaderDict)
    y .-= y[1] #baseline subtract
    real_traces[i] = y
    real_t[i] = t
    first_peak_idx = findfirst(t .>= response_window[1])
    last_peak_idx = findfirst(t .>= response_window[2])
    real_amp[i] = -minimum(y[first_peak_idx:last_peak_idx])
end

dt = real_t[1][2] - real_t[1][1]

#%% 2) Parse stimulus info (this section goes away with ElectroPhysiology update)

stimulus_intensity_levels = map(parse_stimulus_name, real_dataset.HeaderDict["abfPath"])
intensity_levels = map(calculate_photons, real_dataset.HeaderDict["abfPath"])
sorted_idx = sortperm(intensity_levels)

intensity_levels = intensity_levels[sorted_idx]
stimulus_intensity_levels = stimulus_intensity_levels[sorted_idx]
stimulus_model = map(x -> (duration = x.flash_duration, intensity = nd_filter(percent_to_photons_eq(x.percent), x.nd)), stimulus_intensity_levels)

real_dataset.data_array = real_dataset.data_array[sorted_idx, :, :]

#%% 3) Build model + parameter set for BaCl+LAP4-like blocked components
#
# Goal: photoreceptor-dominant response (a-wave focused), while preserving the
# full ERG depth map and cell architecture.
# - ON bipolar signaling blocked (LAP4-like) via TRPM1 conductance.
# - OFF bipolar signaling blocked via iGluR conductance.
# - Müller glia component blocked (BaCl-like) via Kir conductances.
# -----------------------------------------------------------------------------
params_dict = load_all_params(editable = true)

params_dict[:ONBC][:g_TRPM1] = 0.0
params_dict[:OFFBC][:g_iGluR] = 0.0
params_dict[:MULLER][:g_Kir_end] = 0.0
params_dict[:MULLER][:g_Kir_stalk] = 0.0

params = dict_to_namedtuple(params_dict)

pc_coords = square_grid_coords(16)
model, u0 = build_column(
    nPC=16,
    nHC=4,
    nONBC=4,
    onbc_pool_size=4,
    nOFFBC=4,
    nA2=4,
    nGC=1,
    nMG=4,
    pc_coords=pc_coords,
)

println("\nCells in model: ", ordered_cells(model))
println("Total states: ", length(u0))

#%% 4) Run fitting experiment 

println("\n" * "=" ^ 70)
println("Starting a-wave parameter fitting...")
println("=" ^ 70)

result = fit_erg(
    model, u0, params;
    cell_types = [:PHOTO],
    stimuli = stimuli,
    real_t = real_t,
    real_traces = real_traces,
    time_window = response_window,
    tspan = (0.0, 6.0),
    dt = 0.01,
    nm_iterations = 500,
    lbfgs_iterations = 100,
    run_lbfgs = true,
    verbose = true,
)

# ─── 6) Generate outputs ─────────────────────────────────────────────────────

outdir = joinpath(@__DIR__, "output", "awave_optim_fit")

plots = plot_fit_diagnostics(
    result, real_t, real_traces, stimuli;
    time_window = response_window,
    outdir = outdir,
)

datasheet_path = save_fit_datasheet(result, joinpath(outdir, "awave_fit_params.csv"))

println("\n" * "=" ^ 70)
println("Fitting complete!")
println("=" ^ 70)
println("\nConvergence:")
println("  Initial loss: $(result.convergence.initial_loss)")
println("  Final loss:   $(result.convergence.final_loss)")
println("  NM converged: $(result.convergence.nm_converged)")
println("  LBFGS converged: $(result.convergence.lbfgs_converged)")

println("\nOutputs:")
println("  Traces plot:       $(plots.traces)")
println("  Residuals plot:    $(plots.residuals)")
println("  Parameters plot:   $(plots.params)")
if !isempty(plots.correlations)
    println("  Correlations plot: $(plots.correlations)")
end
println("  Datasheet:         $(datasheet_path)")

println("\nTop 10 most certain parameters:")
sorted_unc = sort(result.uncertainty, :certainty, rev=true)
for row in first(eachrow(sorted_unc), 10)
    println("  $(row.parameter): $(round(row.estimate, sigdigits=4)) ± $(round(row.std_error, sigdigits=3)) (certainty=$(round(row.certainty, digits=3)))")
end
