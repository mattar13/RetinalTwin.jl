using Revise
using RetinalTwin
using DifferentialEquations
using Statistics
using CairoMakie

include("retinal_column_plotting.jl")

println("=" ^ 60)
println("RetinalColumnModel Example (Full Build)")
println("=" ^ 60)

#%% --------- Build/load model ---------
params = default_retinal_params()
map_path = joinpath(@__DIR__, "data", "column_map.json")
if isfile(map_path)
    model, u0 = load_mapping(map_path)
else
    pc_coords = square_grid_coords(16)
    model, u0 = build_column(; nPC=16, nHC=4, nONBC=4, nOFFBC=0, nA2=4, nGC=1, nMG=4, pc_coords=pc_coords)
    save_mapping(map_path, model, u0)
end

println("Cells: ", ordered_cells(model))
println("Total states: ", length(u0))

# Dark adaptation
tspan_dark = (0.0, 2000.0)
stim_dark = RetinalTwin.make_uniform_flash_stimulus(photon_flux=0.0)
prob_dark = ODEProblem(model, u0, tspan_dark, (params, stim_dark))
sol_dark = solve(prob_dark, Rodas5(); save_everystep=false, save_start=false, save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol_dark.u[end]

# IR protocol settings
stim_start = 80.0
stim_end = 180.0
intensity_levels = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

fig = plot_cell_layout_3d(
    model;
    z_stim=0.0,
    stimulus_func=RetinalTwin.make_uniform_flash_stimulus(stim_start=stim_start, stim_end=stim_end, photon_flux=1.0),
    stimulus_time=(stim_start + stim_end) / 2,
    stimulus_grid_n=61,
    savepath=joinpath(@__DIR__, "plots", "uniform_flash_ir_curve_column_layout_3d.png"),
)
display(fig)


#%% --------- Run IR sweep ---------
tspan = (0.0, 1200.0)
names = ordered_cells(model)
n_cells = length(names)
nI = length(intensity_levels)
peak_dv = fill(NaN, nI, n_cells)
solutions = Vector{Any}(undef, nI)

baseline_window = (stim_start - 50.0, stim_start)
response_window = (stim_start, stim_end + 400.0)

function finite_mean(x)
    vals = [v for v in x if isfinite(v)]
    return isempty(vals) ? NaN : mean(vals)
end

for (i, photon_flux) in enumerate(intensity_levels)
    selected_stimulus = RetinalTwin.make_uniform_flash_stimulus(
        stim_start=stim_start,
        stim_end=stim_end,
        photon_flux=photon_flux,
    )

    prob = ODEProblem(model, u0, tspan, (params, selected_stimulus))
    @time sol = solve(prob, Rodas5(); tstops=[stim_start, stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)
    solutions[i] = sol

    baseline_idx = findall(t -> baseline_window[1] <= t <= baseline_window[2], sol.t)
    response_idx = findall(t -> response_window[1] <= t <= response_window[2], sol.t)

    for (j, nm) in enumerate(names)
        v = state_trace(sol, model, nm, :V)
        if isempty(baseline_idx) || isempty(response_idx)
            peak_dv[i, j] = NaN
            continue
        end
        baseline = mean(v[baseline_idx])
        peak_dv[i, j] = maximum(abs.(v[response_idx] .- baseline))
    end

    println("Intensity $(photon_flux): retcode=$(sol.retcode), points=$(length(sol.t))")
end

mid_idx = cld(nI, 2)
fig_mid = plot_all_cells_v_ca_release(
    solutions[mid_idx],
    model;
    stim_start=stim_start,
    stim_end=stim_end,
    savepath=joinpath(@__DIR__, "plots", "uniform_flash_ir_reference_all_cells.png"),
)
display(fig_mid)

cell_types = [model.cells[nm].cell_type for nm in names]
present_types = unique(cell_types)
palette = cgrad(:tab10, max(length(present_types), 3), categorical=true)
type_color = Dict(ct => palette[i] for (i, ct) in enumerate(present_types))

fig_ir = Figure(size=(1100, 650))
ax = Axis(
    fig_ir[1, 1],
    xlabel="Flash intensity (photon flux)",
    ylabel="Peak |dV| from baseline (mV)",
    xscale=log10,
    title="Uniform Flash Intensity-Response (per-cell voltage peaks)",
)

for (j, nm) in enumerate(names)
    ct = cell_types[j]
    lines!(ax, intensity_levels, peak_dv[:, j], color=(type_color[ct], 0.25), linewidth=1.0)
end

for ct in present_types
    idx = findall(==(ct), cell_types)
    y = [finite_mean(peak_dv[i, idx]) for i in 1:nI]
    lines!(ax, intensity_levels, y, color=type_color[ct], linewidth=3, label=String(ct))
    scatter!(ax, intensity_levels, y, color=type_color[ct], markersize=10)
end

axislegend(ax, position=:rb)
mkpath(joinpath(@__DIR__, "plots"))
save(joinpath(@__DIR__, "plots", "uniform_flash_ir_curve.png"), fig_ir)
display(fig_ir)

csv_path = joinpath(@__DIR__, "data", "uniform_flash_ir_peaks.csv")
mkpath(dirname(csv_path))
open(csv_path, "w") do io
    println(io, "intensity,cell,cell_type,peak_abs_dv_mV")
    for i in 1:nI
        for (j, nm) in enumerate(names)
            println(io, "$(intensity_levels[i]),$(nm),$(cell_types[j]),$(peak_dv[i, j])")
        end
    end
end
println("Saved IR summary CSV: ", csv_path)
