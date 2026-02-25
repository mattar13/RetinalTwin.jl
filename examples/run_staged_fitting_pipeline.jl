using RetinalTwin
using Random
using DifferentialEquations
using Statistics

println("Running staged fitting demo (synthetic ERG target)...")

# Build a compact but full circuit for fitting experiments.
pc_coords = square_grid_coords(9)
model, u0 = build_column(
    nPC=9,
    nHC=4,
    nONBC=4,
    onbc_pool_size=4,
    nOFFBC=4,
    nA2=4,
    nGC=1,
    nMG=4,
    pc_coords=pc_coords,
)

# Intensity-response panel for fitting.
t = collect(0.0:1.0:350.0)
intensities = [0.1, 0.3, 1.0, 3.0]
blank = zeros(length(t), length(intensities))

default_params = default_retinal_params()
true_params = default_params
true_params = merge(true_params, (
    PHOTORECEPTOR_PARAMS = merge(true_params.PHOTORECEPTOR_PARAMS, (gCa=0.9, gKV=2.3,)),
    ON_BIPOLAR_PARAMS = merge(true_params.ON_BIPOLAR_PARAMS, (g_TRPM1=4.1, tau_S=42.0,)),
    MULLER_PARAMS = merge(true_params.MULLER_PARAMS, (g_Kir_end=55.0,)),
))

synthetic_data = ERGDataSet(intensities, t, blank, 1.0, :mouse, :dark_adapted, 8.0, 50.0)
synthetic_traces = simulate_erg_dataset(model, u0, true_params, synthetic_data)

rng = MersenneTwister(33)
noise = 0.03 .* std(vec(synthetic_traces)) .* randn(rng, size(synthetic_traces))
fit_data = ERGDataSet(intensities, t, synthetic_traces .+ noise, 1.0, :mouse, :dark_adapted, 8.0, 50.0)

result = fit_retinal_twin_staged(model, u0, fit_data; params0=default_params, mode=:efficient, rng=MersenneTwister(8))

outdir = joinpath(@__DIR__, "output", "staged_fit")
plots = plot_fit_diagnostics(fit_data, result; outdir=outdir)
datasheet = save_fit_datasheet(result, joinpath(outdir, "parameter_estimates.csv"))

println("\nStage losses:")
for st in result.stages
    println("  $(st.name): $(round(st.loss, sigdigits=5))")
end
println("\nArtifacts:")
println("  Traces plot: $(plots.traces)")
println("  Residual plot: $(plots.residuals)")
println("  Parameter CI plot: $(plots.params)")
println("  Data sheet: $(datasheet)")
