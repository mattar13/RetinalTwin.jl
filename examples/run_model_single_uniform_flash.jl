using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics

# -----------------------------------------------------------------------------
#%% 1) Build model, load parameters, and dark-adapt
# -----------------------------------------------------------------------------
# load_param_fn = joinpath(@__DIR__, "checkpoints", "fit_checkpoint_nelder_mead_best.csv")
load_param_fn = "C:\\Users\\mtarc\\Julia\\dev\\RetinalTwin.jl\\src\\parameters\\retinal_params.csv"

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
stim_duration = 1.0   # ms
stim_intensity = 2.0 # photon flux
tspan = (0.0, 6.0)
dt = 0.001

t, erg, sol, peak_amp = simulate_erg(model, u0_dark, params;
    stim_duration=stim_duration,
    stim_intensity=stim_intensity,
    tspan=tspan, dt=dt,
    verbose=true,
);

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

# -----------------------------------------------------------------------------
#%% 4) PC calcium & glutamate diagnostics
#     Traces the full chain: V → mCa → Ca_f / Ca_s → buffers → Glu
#     Also plots R_glu_inf(V) to compare the Glu steady-state target vs actual Glu
# -----------------------------------------------------------------------------
pc_names   = [nm for nm in names_ordered if model.cells[nm].cell_type == :PC]
onbc_names = [nm for nm in names_ordered if model.cells[nm].cell_type == :ONBC]
offbc_names= [nm for nm in names_ordered if model.cells[nm].cell_type == :OFFBC]

pc1 = first(pc_names)
fig2 = Figure(size=(1000, 1400))
ca_axes = Axis[]

# Row 1: Voltage (drives everything)
ax1 = Axis(fig2[1, 1], title="PC Voltage", ylabel="V (mV)")
lines!(ax1, sol.t, state_trace(sol, model, pc1, :V), color=:black, linewidth=2)
vlines!(ax1, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
push!(ca_axes, ax1)

# Row 2: Ca channel gating variable
ax2 = Axis(fig2[2, 1], title="Ca Channel Gate (mCa)", ylabel="mCa")
lines!(ax2, sol.t, state_trace(sol, model, pc1, :mCa), color=:royalblue, linewidth=2)
vlines!(ax2, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
push!(ca_axes, ax2)

# Row 3: Fast and slow calcium pools
ax3 = Axis(fig2[3, 1], title="Calcium Pools", ylabel="[Ca]")
lines!(ax3, sol.t, state_trace(sol, model, pc1, :Ca_f), color=:red, linewidth=2, label="Ca_f (fast)")
lines!(ax3, sol.t, state_trace(sol, model, pc1, :Ca_s), color=:orange, linewidth=2, label="Ca_s (slow)")
vlines!(ax3, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax3, position=:rt)
push!(ca_axes, ax3)

# Row 4: Calcium buffers (fast compartment)
ax4 = Axis(fig2[4, 1], title="Ca Buffers (fast compartment)", ylabel="[CaB]")
lines!(ax4, sol.t, state_trace(sol, model, pc1, :CaB_lf), color=:teal, linewidth=2, label="CaB_lf (low-aff)")
lines!(ax4, sol.t, state_trace(sol, model, pc1, :CaB_hf), color=:purple, linewidth=2, label="CaB_hf (high-aff)")
vlines!(ax4, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax4, position=:rt)
push!(ca_axes, ax4)

# Row 5: Calcium buffers (slow compartment)
ax5 = Axis(fig2[5, 1], title="Ca Buffers (slow compartment)", ylabel="[CaB]")
lines!(ax5, sol.t, state_trace(sol, model, pc1, :CaB_ls), color=:teal, linewidth=2, linestyle=:dash, label="CaB_ls (low-aff)")
lines!(ax5, sol.t, state_trace(sol, model, pc1, :CaB_hs), color=:purple, linewidth=2, linestyle=:dash, label="CaB_hs (high-aff)")
vlines!(ax5, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax5, position=:rt)
push!(ca_axes, ax5)

# Row 6: Glutamate — actual vs steady-state target R_glu_inf(V)
ax6 = Axis(fig2[6, 1], title="Glutamate: actual vs R_glu_inf(V) target", ylabel="Glu", xlabel="Time (s)")
glu_trace = state_trace(sol, model, pc1, :Glu)
v_trace = state_trace(sol, model, pc1, :V)
glu_inf_trace = [params.PHOTO.a_Glu * RetinalTwin.R_glu_inf(v, params.PHOTO) for v in v_trace]
lines!(ax6, sol.t, glu_trace, color=:black, linewidth=2, label="Glu (actual)")
lines!(ax6, sol.t, glu_inf_trace, color=:red, linewidth=1.5, linestyle=:dash, label="a_Glu * R_glu_inf(V)")
vlines!(ax6, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax6, position=:rt)
push!(ca_axes, ax6)

for ax in ca_axes[1:end-1]
    hidexdecorations!(ax; grid=false)
end
linkxaxes!(ca_axes...)

fig2

# -----------------------------------------------------------------------------
#%% 5) PC Glu → Bipolar synapse diagnostics
#     Traces: PC Glu → mGluR6 signal (S) → TRPM1 conductance → ONBC V
#             PC Glu → iGluR (A*D) → iGluR conductance → OFFBC V
# -----------------------------------------------------------------------------
onbc1 = first(onbc_names)
offbc1 = first(offbc_names)

fig3 = Figure(size=(1000, 1400))
syn_axes = Axis[]

# Row 1: PC Glutamate release (shared input to both pathways)
ax_glu = Axis(fig3[1, 1:2], title="PC Glutamate Release", ylabel="Glu")
pc_cols = cgrad(:viridis, max(length(pc_names), 2); categorical=true)
for (j, nm) in enumerate(pc_names)
    lines!(ax_glu, sol.t, state_trace(sol, model, nm, :Glu),
        color=pc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_glu, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
length(pc_names) > 1 && axislegend(ax_glu, position=:rb)
push!(syn_axes, ax_glu)

# --- ON Bipolar column (left) ---

# Row 2 left: mGluR6 signal S (inv_hill of Glu → when Glu drops, S rises)
ax_s = Axis(fig3[2, 1], title="ONBC: mGluR6 Signal (S)", ylabel="S")
onbc_cols = cgrad(:blues, max(length(onbc_names), 2); categorical=true)
for (j, nm) in enumerate(onbc_names)
    lines!(ax_s, sol.t, state_trace(sol, model, nm, :S),
        color=onbc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_s, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
length(onbc_names) > 1 && axislegend(ax_s, position=:rb)
push!(syn_axes, ax_s)

# Row 3 left: TRPM1 conductance = g_TRPM1 * S
ax_g_trpm1 = Axis(fig3[3, 1], title="ONBC: TRPM1 Conductance", ylabel="g_TRPM1 * S (nS)")
g_trpm1 = params.ONBC.g_TRPM1
for (j, nm) in enumerate(onbc_names)
    s_trace = state_trace(sol, model, nm, :S)
    lines!(ax_g_trpm1, sol.t, g_trpm1 .* s_trace,
        color=onbc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_g_trpm1, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
push!(syn_axes, ax_g_trpm1)

# Row 4 left: ONBC Voltage
ax_onbc_v = Axis(fig3[4, 1], title="ONBC: Voltage", ylabel="V (mV)", xlabel="Time (s)")
for (j, nm) in enumerate(onbc_names)
    lines!(ax_onbc_v, sol.t, state_trace(sol, model, nm, :V),
        color=onbc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_onbc_v, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
push!(syn_axes, ax_onbc_v)

# --- OFF Bipolar column (right) ---

# Row 2 right: iGluR activation A (hill of Glu)
ax_a = Axis(fig3[2, 2], title="OFFBC: iGluR Activation (A)", ylabel="A")
offbc_cols = cgrad(:reds, max(length(offbc_names), 2); categorical=true)
for (j, nm) in enumerate(offbc_names)
    lines!(ax_a, sol.t, state_trace(sol, model, nm, :A),
        color=offbc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_a, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
length(offbc_names) > 1 && axislegend(ax_a, position=:rb)
push!(syn_axes, ax_a)

# Row 3 right: iGluR conductance = g_iGluR * A * D
ax_g_iglu = Axis(fig3[3, 2], title="OFFBC: iGluR Conductance", ylabel="g_iGluR * A * D (nS)")
g_iglu = params.OFFBC.g_iGluR
for (j, nm) in enumerate(offbc_names)
    a_trace = state_trace(sol, model, nm, :A)
    d_trace = state_trace(sol, model, nm, :D)
    lines!(ax_g_iglu, sol.t, g_iglu .* a_trace .* d_trace,
        color=offbc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_g_iglu, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
push!(syn_axes, ax_g_iglu)

# Row 4 right: OFFBC Voltage
ax_offbc_v = Axis(fig3[4, 2], title="OFFBC: Voltage", ylabel="V (mV)", xlabel="Time (s)")
for (j, nm) in enumerate(offbc_names)
    lines!(ax_offbc_v, sol.t, state_trace(sol, model, nm, :V),
        color=offbc_cols[j], linewidth=1.5, label=String(nm))
end
vlines!(ax_offbc_v, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
push!(syn_axes, ax_offbc_v)

for ax in syn_axes[1:end-2]  # keep xlabel on both bottom axes
    hidexdecorations!(ax; grid=false)
end
linkxaxes!(syn_axes...)

fig3
