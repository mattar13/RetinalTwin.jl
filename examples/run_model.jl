using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics

# -----------------------------------------------------------------------------
#%% 1) Build model, load parameters, and dark-adapt
# -----------------------------------------------------------------------------
load_param_fn = joinpath(@__DIR__, "checkpoints", "fit_checkpoint_nelder_mead_best.csv")
params_dict = load_all_params(csv_path = load_param_fn, editable = true)
params = dict_to_namedtuple(params_dict)

col_path = joinpath(@__DIR__, "structure", "default_column.json")
model, u0 = load_mapping(col_path)

println("Cells in model: ", ordered_cells(model))
println("Total states: ", length(u0))

u0_dark = dark_adapt(model, u0, params; abstol=1e-6, reltol=1e-4)

# -----------------------------------------------------------------------------
#%% 2) Simulate a single ERG response
# -----------------------------------------------------------------------------
stim_duration = 10.0   # ms
stim_intensity = 1000.0 # photon flux
tspan = (0.0, 2.0)
dt = 0.001

t, erg, sol, peak_amp = simulate_erg(model, u0_dark, params;
    stim_duration=stim_duration,
    stim_intensity=stim_intensity,
    tspan=tspan, dt=dt,
    verbose=true,
)

# -----------------------------------------------------------------------------
#%% 3) Plot: ERG (col 1) + cellular voltage components (col 2)
# -----------------------------------------------------------------------------
names_ordered = ordered_cells(model)
present_types = present_cell_types(model; names=names_ordered)

stim_start = 0.0
stim_end = stim_start + stim_duration / 1000.0  # convert ms -> s

nrows = length(present_types)
fig = Figure(size=(1200, max(400, 200 * nrows)))

# Column 1: ERG field potential (spans all rows)
ax_erg = Axis(fig[1:nrows, 1],
    title="ERG Field Potential",
    xlabel="Time (s)",
    ylabel="Amplitude (μV)",
)
lines!(ax_erg, collect(t), erg, color=:black, linewidth=2)
vlines!(ax_erg, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)

# Column 2: per-cell-type voltage traces (one row per type)
v_axes = Axis[]
for (i, ctype) in enumerate(present_types)
    type_names = [nm for nm in names_ordered if model.cells[nm].cell_type == ctype]
    cols = cgrad(:viridis, max(length(type_names), 2); categorical=true)

    ax_v = Axis(fig[i, 2],
        title=i == 1 ? "Cellular Voltage" : "",
        subtitle=String(ctype),
        xlabel=i == nrows ? "Time (s)" : "",
        ylabel="V (mV)",
    )
    for (j, nm) in enumerate(type_names)
        lines!(ax_v, sol.t, state_trace(sol, model, nm, :V),
            color=cols[j], linewidth=1.5, label=String(nm))
    end
    vlines!(ax_v, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
    length(type_names) > 1 && axislegend(ax_v, position=:rb)
    i < nrows && hidexdecorations!(ax_v; grid=false)
    push!(v_axes, ax_v)
end
linkxaxes!(v_axes...)

fig
