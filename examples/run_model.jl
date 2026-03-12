using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics

# -----------------------------------------------------------------------------
#%% 1) Input file paths (all from examples/inputs/)
# -----------------------------------------------------------------------------
input_dir = joinpath(@__DIR__, "inputs", "default")
structure_fn   = joinpath(input_dir, "default_column.json")
params_fn      = joinpath(input_dir, "retinal_params.csv")
stimulus_fn    = joinpath(input_dir, "stimulus_table.csv")
depth_fn       = joinpath(input_dir, "erg_depth_map.csv")

# -----------------------------------------------------------------------------
#%% 2) Simulate ERG (loads files, dark-adapts, runs all stimuli automatically)
# -----------------------------------------------------------------------------
tspan = (0.0, 6.0)
dt = 0.001

t, erg_traces, solutions, peak_amps = simulate_erg(;
    structure = structure_fn,
    params = params_fn,
    stimulus_table = stimulus_fn,
    depth_csv = depth_fn,
    tspan = tspan, dt = dt,
    verbose = true,
)

# Reload structure and params for plotting diagnostics
model, u0 = load_mapping(structure_fn)
params = load_all_params(csv_path = params_fn)
stimulus_table = load_stimulus_table(stimulus_fn)
sol = solutions[1]  # use first stimulus for detailed diagnostics

# -----------------------------------------------------------------------------
#%% 3) Plot: ERG (col 1) + cellular voltage components (col 2)
# -----------------------------------------------------------------------------
names_ordered = ordered_cells(model)
present_types = present_cell_types(model; names=names_ordered)
n_sweeps = length(stimulus_table)

nrows = length(present_types)
fig = Figure(size=(1200, max(400, 200 * nrows)))

# Shared intensity colormap for all panels
intensity_levels = [s.intensity for s in stimulus_table]
log_I = log10.(intensity_levels)
logI_min, logI_max = extrema(log_I)
trace_cmap = cgrad(:viridis)
sweep_colors = [get(trace_cmap, (li - logI_min) / (logI_max - logI_min + eps())) for li in log_I]

# Column 1: ERG field potential (spans all rows) — overlay all sweeps
ax_erg = Axis(fig[1:nrows, 1],
    title="ERG Field Potential",
    xlabel="Time (s)",
    ylabel="Amplitude (μV)",
)
for (i, s) in enumerate(stimulus_table)
    lines!(ax_erg, collect(t), erg_traces[i], color=sweep_colors[i], linewidth=2)
end
Colorbar(fig[1:nrows, 3], colormap=:viridis, limits=(logI_min, logI_max),
    label="log10 intensity")

# Column 2: per-cell-type voltage (first cell of each type, all sweeps overlaid)
v_axes = Axis[]
for (r, ctype) in enumerate(present_types)
    type_names = [nm for nm in names_ordered if model.cells[nm].cell_type == ctype]
    rep_cell = first(type_names)

    ax_v = Axis(fig[r, 2],
        title=r == 1 ? "Cellular Voltage" : "",
        subtitle=String(ctype) * " ($(rep_cell))",
        xlabel=r == nrows ? "Time (s)" : "",
        ylabel="V (mV)",
    )
    for (i, sol_i) in enumerate(solutions)
        lines!(ax_v, sol_i.t, state_trace(sol_i, model, rep_cell, :V),
            color=sweep_colors[i], linewidth=1.5)
    end
    r < nrows && hidexdecorations!(ax_v; grid=false)
    push!(v_axes, ax_v)
end
linkxaxes!(v_axes...)

fig

# -----------------------------------------------------------------------------
#%% 4) PC calcium & glutamate diagnostics
#     Traces the full chain: V → mCa → Ca_f / Ca_s → buffers → Glu
#     All sweeps overlaid, color-coded by intensity
# -----------------------------------------------------------------------------
pc_names   = [nm for nm in names_ordered if model.cells[nm].cell_type == :PC]
onbc_names = [nm for nm in names_ordered if model.cells[nm].cell_type == :ONBC]
offbc_names= [nm for nm in names_ordered if model.cells[nm].cell_type == :OFFBC]

pc1 = first(pc_names)
fig2 = Figure(size=(1100, 1400))
ca_axes = Axis[]

# Row 1: Voltage (drives everything)
ax1 = Axis(fig2[1, 1], title="PC Voltage ($(pc1))", ylabel="V (mV)")
for (i, sol_i) in enumerate(solutions)
    lines!(ax1, sol_i.t, state_trace(sol_i, model, pc1, :V), color=sweep_colors[i], linewidth=1.5)
end
push!(ca_axes, ax1)

# Row 2: Ca channel gating variable
ax2 = Axis(fig2[2, 1], title="Ca Channel Gate (mCa)", ylabel="mCa")
for (i, sol_i) in enumerate(solutions)
    lines!(ax2, sol_i.t, state_trace(sol_i, model, pc1, :mCa), color=sweep_colors[i], linewidth=1.5)
end
push!(ca_axes, ax2)

# Row 3: Fast and slow calcium pools (solid = Ca_f, dashed = Ca_s)
ax3 = Axis(fig2[3, 1], title="Calcium Pools (solid=Ca_f, dashed=Ca_s)", ylabel="[Ca]")
for (i, sol_i) in enumerate(solutions)
    lines!(ax3, sol_i.t, state_trace(sol_i, model, pc1, :Ca_f), color=sweep_colors[i], linewidth=1.5)
    lines!(ax3, sol_i.t, state_trace(sol_i, model, pc1, :Ca_s), color=sweep_colors[i], linewidth=1.5, linestyle=:dash)
end
push!(ca_axes, ax3)

# Row 4: Calcium buffers — fast compartment (solid = low-aff, dashed = high-aff)
ax4 = Axis(fig2[4, 1], title="Ca Buffers fast (solid=low, dashed=high)", ylabel="[CaB]")
for (i, sol_i) in enumerate(solutions)
    lines!(ax4, sol_i.t, state_trace(sol_i, model, pc1, :CaB_lf), color=sweep_colors[i], linewidth=1.5)
    lines!(ax4, sol_i.t, state_trace(sol_i, model, pc1, :CaB_hf), color=sweep_colors[i], linewidth=1.5, linestyle=:dash)
end
push!(ca_axes, ax4)

# Row 5: Calcium buffers — slow compartment
ax5 = Axis(fig2[5, 1], title="Ca Buffers slow (solid=low, dashed=high)", ylabel="[CaB]")
for (i, sol_i) in enumerate(solutions)
    lines!(ax5, sol_i.t, state_trace(sol_i, model, pc1, :CaB_ls), color=sweep_colors[i], linewidth=1.5)
    lines!(ax5, sol_i.t, state_trace(sol_i, model, pc1, :CaB_hs), color=sweep_colors[i], linewidth=1.5, linestyle=:dash)
end
push!(ca_axes, ax5)

# Row 6: Glutamate (solid = actual, dashed = R_glu_inf target)
ax6 = Axis(fig2[6, 1], title="Glutamate (solid=actual, dashed=R_glu_inf target)", ylabel="Glu", xlabel="Time (s)")
for (i, sol_i) in enumerate(solutions)
    glu_trace = state_trace(sol_i, model, pc1, :Glu)
    v_trace = state_trace(sol_i, model, pc1, :V)
    glu_inf_trace = [params.PHOTO.a_Glu * RetinalTwin.R_glu_inf(v, params.PHOTO) for v in v_trace]
    lines!(ax6, sol_i.t, glu_trace, color=sweep_colors[i], linewidth=1.5)
    lines!(ax6, sol_i.t, glu_inf_trace, color=sweep_colors[i], linewidth=1.0, linestyle=:dash)
end
push!(ca_axes, ax6)

for ax in ca_axes[1:end-1]
    hidexdecorations!(ax; grid=false)
end
linkxaxes!(ca_axes...)
Colorbar(fig2[1:6, 2], colormap=:viridis, limits=(logI_min, logI_max), label="log10 intensity")

fig2

# -----------------------------------------------------------------------------
#%% 5) PC Glu → Bipolar synapse diagnostics (all sweeps overlaid)
#     Traces: PC Glu → mGluR6 signal (S) → TRPM1 conductance → ONBC V
#             PC Glu → iGluR (A*D) → iGluR conductance → OFFBC V
# -----------------------------------------------------------------------------
onbc1 = first(onbc_names)
offbc1 = first(offbc_names)

fig3 = Figure(size=(1100, 1400))
syn_axes = Axis[]

# Row 1: PC Glutamate release (first PC, all sweeps)
ax_glu = Axis(fig3[1, 1:2], title="PC Glutamate Release ($(pc1))", ylabel="Glu")
for (i, sol_i) in enumerate(solutions)
    lines!(ax_glu, sol_i.t, state_trace(sol_i, model, pc1, :Glu),
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_glu)

# --- ON Bipolar column (left) ---

# Row 2 left: mGluR6 signal S
ax_s = Axis(fig3[2, 1], title="ONBC: mGluR6 Signal S ($(onbc1))", ylabel="S")
for (i, sol_i) in enumerate(solutions)
    lines!(ax_s, sol_i.t, state_trace(sol_i, model, onbc1, :S),
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_s)

# Row 3 left: TRPM1 conductance = g_TRPM1 * S
ax_g_trpm1 = Axis(fig3[3, 1], title="ONBC: TRPM1 Conductance", ylabel="g_TRPM1 * S (nS)")
g_trpm1 = params.ONBC.g_TRPM1
for (i, sol_i) in enumerate(solutions)
    s_trace = state_trace(sol_i, model, onbc1, :S)
    lines!(ax_g_trpm1, sol_i.t, g_trpm1 .* s_trace,
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_g_trpm1)

# Row 4 left: ONBC Voltage
ax_onbc_v = Axis(fig3[4, 1], title="ONBC: Voltage ($(onbc1))", ylabel="V (mV)", xlabel="Time (s)")
for (i, sol_i) in enumerate(solutions)
    lines!(ax_onbc_v, sol_i.t, state_trace(sol_i, model, onbc1, :V),
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_onbc_v)

# --- OFF Bipolar column (right) ---

# Row 2 right: iGluR activation A
ax_a = Axis(fig3[2, 2], title="OFFBC: iGluR Activation A ($(offbc1))", ylabel="A")
for (i, sol_i) in enumerate(solutions)
    lines!(ax_a, sol_i.t, state_trace(sol_i, model, offbc1, :A),
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_a)

# Row 3 right: iGluR conductance = g_iGluR * A * D
ax_g_iglu = Axis(fig3[3, 2], title="OFFBC: iGluR Conductance", ylabel="g_iGluR * A * D (nS)")
g_iglu = params.OFFBC.g_iGluR
for (i, sol_i) in enumerate(solutions)
    a_trace = state_trace(sol_i, model, offbc1, :A)
    d_trace = state_trace(sol_i, model, offbc1, :D)
    lines!(ax_g_iglu, sol_i.t, g_iglu .* a_trace .* d_trace,
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_g_iglu)

# Row 4 right: OFFBC Voltage
ax_offbc_v = Axis(fig3[4, 2], title="OFFBC: Voltage ($(offbc1))", ylabel="V (mV)", xlabel="Time (s)")
for (i, sol_i) in enumerate(solutions)
    lines!(ax_offbc_v, sol_i.t, state_trace(sol_i, model, offbc1, :V),
        color=sweep_colors[i], linewidth=1.5)
end
push!(syn_axes, ax_offbc_v)

for ax in syn_axes[1:end-2]  # keep xlabel on both bottom axes
    hidexdecorations!(ax; grid=false)
end
linkxaxes!(syn_axes...)
Colorbar(fig3[1:4, 3], colormap=:viridis, limits=(logI_min, logI_max), label="log10 intensity")

fig3
