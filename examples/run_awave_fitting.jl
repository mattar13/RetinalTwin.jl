using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics

println("=" ^ 70)
println("A-wave simulation workflow (photoreceptor-dominant BaCl + LAP4)")
println("=" ^ 70)

# -----------------------------------------------------------------------------
# 1) Experiment metadata
# -----------------------------------------------------------------------------
erg_file = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods\nd0_1p_2ms.abf"
println("Target ABF file for upcoming fitting workflow:\n  ", erg_file)

# -----------------------------------------------------------------------------
# 2) Build model + parameter set for BaCl+LAP4-like blocked components
#
# Goal: photoreceptor-dominant response (a-wave focused), while preserving the
# full ERG depth map and cell architecture.
# - ON bipolar signaling blocked (LAP4-like) via TRPM1 conductance.
# - OFF bipolar signaling blocked via iGluR conductance.
# - Müller glia component blocked (BaCl-like) via Kir conductances.
# -----------------------------------------------------------------------------
params_dict = default_retinal_params(editable=true)

# ON bipolar block (b-wave component)
params_dict[:ON_BIPOLAR_PARAMS][:g_TRPM1] = 0.0

# OFF bipolar block
params_dict[:OFF_BIPOLAR_PARAMS][:g_iGluR] = 0.0

# Müller glia block (BaCl-like)
params_dict[:MULLER_PARAMS][:g_Kir_end] = 0.0
params_dict[:MULLER_PARAMS][:g_Kir_stalk] = 0.0

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

println("Cells in model: ", ordered_cells(model))
println("Total states: ", length(u0))

# -----------------------------------------------------------------------------
# 3) Dark adapt initial condition
# -----------------------------------------------------------------------------
tspan_dark = (0.0, 2000.0)
stim_dark = RetinalTwin.make_uniform_flash_stimulus(photon_flux=0.0)
prob_dark = ODEProblem(model, u0, tspan_dark, (params, stim_dark))
sol_dark = solve(
    prob_dark,
    Rodas5();
    save_everystep=false,
    save_start=false,
    save_end=true,
    abstol=1e-6,
    reltol=1e-4,
)
u0_dark = sol_dark.u[end]

# -----------------------------------------------------------------------------
# 4) IR protocol and ERG extraction for each flash intensity
# -----------------------------------------------------------------------------
stim_start = 80.0
stim_end = 82.0
intensity_levels = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# save outputs
solutions = Vector{Any}(undef, length(intensity_levels))
erg_traces = Vector{Vector{Float64}}(undef, length(intensity_levels))
peak_awave = fill(NaN, length(intensity_levels))

tspan = (0.0, 350.0)
response_window = (stim_start, stim_start + 120.0)

for (i, photon_flux) in enumerate(intensity_levels)
    stim = RetinalTwin.make_uniform_flash_stimulus(
        stim_start=stim_start,
        stim_end=stim_end,
        photon_flux=photon_flux,
    )

    prob = ODEProblem(model, u0_dark, tspan, (params, stim))
    sol = solve(
        prob,
        Rodas5();
        tstops=[stim_start, stim_end],
        saveat=0.5,
        abstol=1e-6,
        reltol=1e-4,
    )

    t_erg, erg = compute_field_potential(model, params, sol)
    solutions[i] = sol
    erg_traces[i] = erg

    response_idx = findall(t -> response_window[1] <= t <= response_window[2], t_erg)
    peak_awave[i] = isempty(response_idx) ? NaN : minimum(erg[response_idx])

    println("Intensity $(photon_flux): retcode=$(sol.retcode), a-wave peak=$(round(peak_awave[i], digits=4))")
end

# -----------------------------------------------------------------------------
# 5) Plot ERG traces (a-wave) across intensities
# -----------------------------------------------------------------------------
fig = Figure(size=(1100, 780))
ax_trace = Axis(
    fig[1, 1],
    xlabel="Time (ms)",
    ylabel="ERG (a.u.)",
    title="Simulated BaCl + LAP4 ERG (parameter-blocked b-wave + MGC components)",
)

palette = cgrad(:viridis, length(intensity_levels), categorical=true)
t_ref, _ = compute_field_potential(model, params, solutions[1])
for (i, I) in enumerate(intensity_levels)
    lines!(ax_trace, t_ref, erg_traces[i], color=palette[i], linewidth=2.0, label="I=$(I)")
end
vspan!(ax_trace, stim_start, stim_end, color=(:gold, 0.15))
axislegend(ax_trace, position=:rb)

ax_ir = Axis(
    fig[2, 1],
    xlabel="Flash intensity (photon flux)",
    ylabel="A-wave peak (minimum ERG)",
    xscale=log10,
    title="A-wave intensity-response",
)
lines!(ax_ir, intensity_levels, peak_awave, color=:black, linewidth=2)
scatter!(ax_ir, intensity_levels, peak_awave, color=:crimson, markersize=10)

mkpath(joinpath(@__DIR__, "plots"))
plot_path = joinpath(@__DIR__, "plots", "awave_ir_simulation.png")
save(plot_path, fig)
display(fig)
println("Saved plot: ", plot_path)

# -----------------------------------------------------------------------------
# 6) Save summary table for future fitting step
# -----------------------------------------------------------------------------
out_csv = joinpath(@__DIR__, "data", "awave_ir_simulation.csv")
open(out_csv, "w") do io
    println(io, "intensity,awave_peak_min_erg")
    for (I, peak) in zip(intensity_levels, peak_awave)
        println(io, "$(I),$(peak)")
    end
end
println("Saved IR summary CSV: ", out_csv)

println("\nNext step: use ElectroPhysiology.jl to read the ABF trace and fit these simulated curves.")
