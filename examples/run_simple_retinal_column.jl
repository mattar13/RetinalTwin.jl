# ============================================================
# simple_retinal_column.jl
# Simple example using retinal_column_model! for photoreceptor → ON bipolar coupling
# ============================================================
using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie


println("=" ^ 60)
println("Simple Retinal Column: Photoreceptor → ON Bipolar")
println("=" ^ 60)

# ── 1. Load parameters ───────────────────────────────────────
println("\n1. Loading parameters...")
params = default_retinal_params()
println("  ✓ Photoreceptor kHYDRO: $(params.PHOTORECEPTOR_PARAMS.kHYDRO)")
println("  ✓ Photoreceptor parameters: $(length(fieldnames(typeof(params.PHOTORECEPTOR_PARAMS)))) fields")
println("  ✓ ON bipolar parameters: $(length(fieldnames(typeof(params.ON_BIPOLAR_PARAMS)))) fields")

# ── 2. Build initial conditions ──────────────────────────────
println("\n2. Building initial conditions...")
u0 = retinal_column_initial_conditions(params)

println("  ✓ Total state variables: $(length(u0))")

# ── 3. Define stimulus ────────────────────────────────────────
println("\n3. Computing steady state (dark, no stimulus)...")
#Create an empty stimulus function
stim_func_dark(t) = RetinalTwin.single_flash(t;photon_flux = 0.0)

tspan = (0.0, 1000.0)
prob = ODEProblem(retinal_column_model!, u0, tspan, (params, stim_func_dark))
sol = solve(prob, Rodas5(); save_everystep=false, save_start=false,save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol.u[end]

stim_start = 80.0   # ms
stim_end = 180.0      # ms (5 ms flash)
photon_flux = 50.0  # photons/µm²/ms
stim_func(t) = RetinalTwin.single_flash(t; stim_start = stim_start, stim_end = stim_end, photon_flux = photon_flux)
println("\n4. Stimulus configuration:")
println("  ✓ Intensity: $(photon_flux) ph/µm²/ms")
println("  ✓ Duration: $(stim_start) - $(stim_end) ms")

# ── 4. Solve ODE system ───────────────────────────────────────
println("\n5. Solving coupled system...")

tspan = (0.0, 1000.0)  # ms
prob = ODEProblem(retinal_column_model!, u0, tspan, (params, stim_func))

sol = solve(prob, Rodas5(); tstops = [stim_start, stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)
println("  ✓ Solver status: $(sol.retcode)")
println("  ✓ Time points: $(length(sol.t))")

# ── 5. Extract results ────────────────────────────────────────
println("\n6. Extracting results...")

t = sol.t
t_series = range(0.0, sol.t[end], length=10000)

# Photoreceptor variables
V_photo = map(t-> sol(t)[20], t_series)
Ca_photo = map(t-> sol(t)[15], t_series)
Glu_photo = map(t-> sol(t)[21], t_series)  # Glutamate is now a state variable!

# ON bipolar variables
V_onbc = map(t-> sol(t)[22], t_series)  # V
Ca_onbc = map(t-> sol(t)[25], t_series)  # Ca
Glu_onbc = map(t-> sol(t)[27], t_series)  # ON BC glutamate release

V_offbc = map(t-> sol(t)[28], t_series)  # OFF BC V
Ca_offbc = map(t-> sol(t)[31], t_series)  # OFF BC Ca
A_offbc = map(t-> sol(t)[32], t_series)  # OFF BC iGluR activation
D_offbc = map(t-> sol(t)[33], t_series)  # OFF BC desensitization
Glu_offbc = map(t-> sol(t)[34], t_series)  # OFF BC glutamate release

V_a2 = map(t-> sol(t)[35], t_series)  # A2 amacrine V
Ca_a2 = map(t-> sol(t)[38], t_series)  # A2 amacrine Ca
Gly_a2 = map(t-> sol(t)[41], t_series)  # A2 amacrine glycine release
V_gc = map(t-> sol(t)[42], t_series)  # Ganglion V
# Ganglion model currently has no explicit Ca state; use excitatory gate as temporary proxy.
Ca_gc = map(t-> sol(t)[46], t_series)  # Ganglion Ca proxy (sE)
gE_gc = map(t-> sol(t)[46], t_series)  # Ganglion excitatory gate
gI_gc = map(t-> sol(t)[47], t_series)  # Ganglion inhibitory gate

# Muller glia variables (expected as 4 states appended after ganglion).
V_muller = map(t-> sol(t)[48], t_series)  # Muller glia V
K_end_muller = map(t-> sol(t)[49], t_series)  # Muller glia K_o end
Glu_muller = map(t-> sol(t)[51], t_series)  # Muller glia glutamate release

#%% ── 6. Visualization ──────────────────────────────────────────
println("\n7. Creating plots...")

fig1 = Figure(size=(1400, 1200))

row_labels = ["Photoreceptor", "ON Bipolar", "Ganglion", "Muller Glia"]
voltage_data = [V_photo, V_onbc, V_gc, V_muller]
calcium_data = [Ca_photo, Ca_onbc, Ca_gc, K_end_muller]
release_data = [Glu_photo, Glu_onbc, nothing, Glu_muller]

voltage_axes = Axis[]
calcium_axes = Axis[]
release_axes = Axis[]

for i in eachindex(row_labels)
    ax_v = Axis(
        fig1[i, 1],
        title=i == 1 ? "Voltage" : "",
        subtitle=row_labels[i],
        xlabel=i == length(row_labels) ? "Time (ms)" : "",
        ylabel="V (mV)",
    )
    lines!(ax_v, t_series, voltage_data[i], color=:blue, linewidth=2)
    vlines!(ax_v, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
    i < length(row_labels) && hidexdecorations!(ax_v, grid=false)
    push!(voltage_axes, ax_v)

    ax_ca = Axis(
        fig1[i, 2],
        title=i == 1 ? "Calcium" : "",
        subtitle=row_labels[i],
        xlabel=i == length(row_labels) ? "Time (ms)" : "",
        ylabel="Ca (µM)",
    )
    row_labels[i] == "Muller Glia" && (ax_ca.ylabel = "K_o end (mM)")
    lines!(ax_ca, t_series, calcium_data[i], color=:purple, linewidth=2)
#     ylims!(ax_ca, 0.0, 1.0)
    vlines!(ax_ca, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
    i < length(row_labels) && hidexdecorations!(ax_ca, grid=false)
    push!(calcium_axes, ax_ca)

    ax_rel = Axis(
        fig1[i, 3],
        title=i == 1 ? "Release / Synaptic Output" : "",
        subtitle=row_labels[i],
        xlabel=i == length(row_labels) ? "Time (ms)" : "",
        ylabel=row_labels[i] == "A2 Amacrine" ? "Glycine (µM)" : row_labels[i] == "Ganglion" ? "Gate (a.u.)" : "Glutamate (µM)",
    )
    if row_labels[i] == "Ganglion"
        lines!(ax_rel, t_series, gE_gc, color=:red, linewidth=2, label="sE")
        lines!(ax_rel, t_series, gI_gc, color=:orange, linewidth=2, label="sI")
        axislegend(ax_rel, position=:rt)
        ylims!(ax_rel, 0.0, 1.0)
    else
        lines!(ax_rel, t_series, release_data[i], color=:green, linewidth=2)
    end
    ylims!(ax_rel, 0.0, 1.0)
    vlines!(ax_rel, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
    i < length(row_labels) && hidexdecorations!(ax_rel, grid=false)
    push!(release_axes, ax_rel)
end

linkxaxes!(voltage_axes...)
linkxaxes!(calcium_axes...)
linkxaxes!(release_axes...)
save("examples/plots/simple_retinal_column.png", fig1)


#Voltage vs glutamate release
C_rng = range(0.0, 1.0, length=100)
G_rng = [RetinalTwin.R_inf(c, params.ON_BIPOLAR_PARAMS.K_Release, params.ON_BIPOLAR_PARAMS.n_Release) for c in C_rng]
fig2 = Figure(size=(500, 500))
ax3a = Axis(fig2[1, 1],
           xlabel="Ca (µM)",
           ylabel="Glutamate Release (µM)",
           title="Glutamate Release vs Ca")
lines!(ax3a, C_rng, G_rng, color=:blue, linewidth=2)
ylims!(ax3a, 0.0, 1.0)
save("examples/plots/voltage_vs_glutamate_release.png", fig2)


# Photoreceptor voltage vs glutamate release percentage
V_glu_rng = range(-70.0, 0.0, length=400)
R_glu_curve = @. params.PHOTORECEPTOR_PARAMS.alpha_Glu / (1.0 + exp(-(V_glu_rng - params.PHOTORECEPTOR_PARAMS.V_Glu_half) / params.PHOTORECEPTOR_PARAMS.V_Glu_slope))
R_glu_pct_curve = @. 100.0 * R_glu_curve / params.PHOTORECEPTOR_PARAMS.alpha_Glu
R_glu_pct_trace = @. 100.0 * Glu_photo / params.PHOTORECEPTOR_PARAMS.alpha_Glu

fig_photo_glu = Figure(size=(700, 500))
ax_photo_glu = Axis(
    fig_photo_glu[1, 1],
    xlabel="Photoreceptor V (mV)",
    ylabel="R_Glu (% of max)",
    title="Photoreceptor Voltage vs R_Glu",
)
lines!(ax_photo_glu, V_glu_rng, R_glu_pct_curve, color=:blue, linewidth=3, label="R_Glu(V) analytic")
scatter!(ax_photo_glu, V_photo, R_glu_pct_trace, color=(:orange, 0.35), markersize=4, label="Simulation samples")
xlims!(ax_photo_glu, -70.0, 0.0)
ylims!(ax_photo_glu, 0.0, 100.0)
axislegend(ax_photo_glu, position=:rb)
save("examples/plots/photoreceptor_voltage_vs_r_glu_percent.png", fig_photo_glu)


# ON bipolar CaL gating phase plot: m_inf(V) with trajectory overlay
V_range = range(-70.0, 0.0, length=400)
m_inf_curve = @. 1.0 / (1.0 + exp(-(V_range - params.ON_BIPOLAR_PARAMS.Vm_half) / params.ON_BIPOLAR_PARAMS.km_slope))
m_inf_onbc = @. 1.0 / (1.0 + exp(-(V_onbc - params.ON_BIPOLAR_PARAMS.Vm_half) / params.ON_BIPOLAR_PARAMS.km_slope))

fig_phase = Figure(size=(700, 500))
ax_phase = Axis(
    fig_phase[1, 1],
    xlabel="V (mV)",
    ylabel="m_inf (CaL)",
    title="ON Bipolar CaL Gating vs Voltage",
)
lines!(ax_phase, V_range, m_inf_curve, color=:blue, linewidth=3, label="m_inf(V) analytic")
scatter!(ax_phase, V_onbc, m_inf_onbc, color=(:orange, 0.35), markersize=4, label="Trajectory samples")
xlims!(ax_phase, -70.0, 0.0)
ylims!(ax_phase, 0.0, 1.0)
axislegend(ax_phase, position=:rb)
save("examples/plots/on_bipolar_calf_minf_phase.png", fig_phase)


# ON bipolar current/Ca-in diagnostics
I_CaL_onbc = @. params.ON_BIPOLAR_PARAMS.g_CaL * m_inf_onbc * (V_onbc - params.ON_BIPOLAR_PARAMS.E_Ca)
Ca_in_onbc = @. max(-I_CaL_onbc, 0.0)

fig_onbc_diag = Figure(size=(900, 700))
ax_ic = Axis(
    fig_onbc_diag[1, 1],
    xlabel="Time (ms)",
    ylabel="I_CaL (nA)",
    title="ON Bipolar I_CaL(t)",
)
lines!(ax_ic, t_series, I_CaL_onbc, color=:blue, linewidth=2)
vlines!(ax_ic, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)

ax_cain = Axis(
    fig_onbc_diag[2, 1],
    xlabel="Time (ms)",
    ylabel="Ca_in (a.u.)",
    title="ON Bipolar Ca_in(t) = max(-I_CaL, 0)",
)
lines!(ax_cain, t_series, Ca_in_onbc, color=:orange, linewidth=2)
vlines!(ax_cain, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)

linkxaxes!(ax_ic, ax_cain)
save("examples/plots/on_bipolar_ical_cain_time.png", fig_onbc_diag)


# ON bipolar mGluR6 activation vs photoreceptor glutamate
glu_max = max(1.0, maximum(Glu_photo))
glu_rng = range(0.0, glu_max, length=400)
S_inf_curve = @. 1.0 / (1.0 + (glu_rng / params.ON_BIPOLAR_PARAMS.K_Glu)^params.ON_BIPOLAR_PARAMS.n_Glu)
S_inf_trace = @. 1.0 / (1.0 + (max(Glu_photo, 0.0) / params.ON_BIPOLAR_PARAMS.K_Glu)^params.ON_BIPOLAR_PARAMS.n_Glu)

fig_sinf = Figure(size=(700, 500))
ax_sinf = Axis(
    fig_sinf[1, 1],
    xlabel="Photoreceptor Glutamate (µM)",
    ylabel="S_INF (a.u.)",
    title="ON Bipolar mGluR6 Activation vs Glutamate",
)
lines!(ax_sinf, glu_rng, S_inf_curve, color=:blue, linewidth=3, label="S_INF(glu) analytic")
scatter!(ax_sinf, Glu_photo, S_inf_trace, color=(:orange, 0.35), markersize=4, label="Simulation samples")
xlims!(ax_sinf, 0.0, glu_max)
ylims!(ax_sinf, 0.0, 1.0)
axislegend(ax_sinf, position=:rb)
save("examples/plots/on_bipolar_mglur6_sinf_vs_glu.png", fig_sinf)


# ON bipolar glutamate release vs RGC excitatory synaptic activation (sE)
glu_onbc_max = max(1.0, maximum(Glu_onbc))
glu_onbc_rng = range(0.0, glu_onbc_max, length=400)
sE_inf_curve = @. (glu_onbc_rng^params.GANGLION_PARAMS.n_preE) / (params.GANGLION_PARAMS.K_preE^params.GANGLION_PARAMS.n_preE + glu_onbc_rng^params.GANGLION_PARAMS.n_preE + eps())
sE_ss_curve = @. params.GANGLION_PARAMS.a_preE * sE_inf_curve

fig_onbc_to_sE = Figure(size=(700, 500))
ax_onbc_to_sE = Axis(
    fig_onbc_to_sE[1, 1],
    xlabel="ON Bipolar Glutamate Release (µM)",
    ylabel="RGC sE (a.u.)",
    title="ON Bipolar Glutamate vs RGC Excitatory Synapse Activation",
)
lines!(ax_onbc_to_sE, glu_onbc_rng, sE_ss_curve, color=:blue, linewidth=3, label="Steady-state target")
scatter!(ax_onbc_to_sE, Glu_onbc, gE_gc, color=(:orange, 0.35), markersize=4, label="Simulation samples")
xlims!(ax_onbc_to_sE, 0.0, glu_onbc_max)
ylims!(ax_onbc_to_sE, 0.0, 1.0)
axislegend(ax_onbc_to_sE, position=:rb)
save("examples/plots/on_bipolar_glu_vs_rgc_sE.png", fig_onbc_to_sE)


# ON bipolar voltage vs glutamate release with simulation overlay
V_onbc_rng = range(-70.0, 0.0, length=400)
m_inf_onbc_curve = @. 1.0 / (1.0 + exp(-(V_onbc_rng - params.ON_BIPOLAR_PARAMS.Vm_half) / params.ON_BIPOLAR_PARAMS.km_slope))
I_CaL_onbc_curve = @. params.ON_BIPOLAR_PARAMS.g_CaL * m_inf_onbc_curve * (V_onbc_rng - params.ON_BIPOLAR_PARAMS.E_Ca)
Ca_in_onbc_curve = @. max(-I_CaL_onbc_curve, 0.0)
R_rel_onbc_curve = @. params.ON_BIPOLAR_PARAMS.a_Release * RetinalTwin.R_inf(Ca_in_onbc_curve, params.ON_BIPOLAR_PARAMS.K_Release, params.ON_BIPOLAR_PARAMS.n_Release)

fig_onbc_v_glu = Figure(size=(700, 500))
ax_onbc_v_glu = Axis(
    fig_onbc_v_glu[1, 1],
    xlabel="ON Bipolar V (mV)",
    ylabel="ON Bipolar Glutamate Release (a.u.)",
    title="ON Bipolar Voltage vs Glutamate Release",
)
lines!(ax_onbc_v_glu, V_onbc_rng, R_rel_onbc_curve, color=:blue, linewidth=3, label="Analytic proxy")
scatter!(ax_onbc_v_glu, V_onbc, Glu_onbc, color=(:orange, 0.35), markersize=4, label="Simulation samples")
xlims!(ax_onbc_v_glu, -70.0, 0.0)
ylims!(ax_onbc_v_glu, 0.0, 1.0)
axislegend(ax_onbc_v_glu, position=:rb)
save("examples/plots/on_bipolar_voltage_vs_glutamate_release.png", fig_onbc_v_glu)


#Plot the desensitization of the OFF bipolar iGluR
fig3 = Figure(size=(500, 500))
ax1a = Axis(fig3[1, 1],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="OFF Bipolar Membrane Potential")
lines!(ax1a, t_series, V_offbc, color=:black, linewidth=2)
vlines!(ax1a, [stim_start, stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

ax2a = Axis(fig3[2, 1],
           xlabel="Time (ms)",
           ylabel="Activation",
           title="AMPA/KAR-like activation")
lines!(ax2a, t_series, A_offbc, color=:green, linewidth=2)
vlines!(ax2a, [stim_start, stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax3a = Axis(fig3[3, 1],
           xlabel="Time (ms)",
           ylabel="Desensitization",
           title="AMPA/KAR-like desensitization")
lines!(ax3a, t_series, D_offbc, color=:red, linewidth=2)
vlines!(ax3a, [stim_start, stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax4a = Axis(fig3[4, 1],
           xlabel="Time (ms)",
           ylabel="Glutamate Release (µM)",
           title="OFF Bipolar Glutamate Release")
lines!(ax4a, t_series, A_offbc .* D_offbc, color=:orange, linewidth=2)
ylims!(ax4a, 0.0, 1.0)
vlines!(ax4a, [stim_start, stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

save("examples/plots/desensitization_off_bipolar_iGluR.png", fig3)

# OFF pathway overview (OFF bipolar + A2 amacrine)
fig_off = Figure(size=(900, 900))
off_series = [
    ("OFF Bipolar V", V_offbc, "V (mV)", :black),
    ("OFF Bipolar Glu Release", Glu_offbc, "Glu (a.u.)", :orange),
    ("A2 Amacrine V", V_a2, "V (mV)", :purple),
    ("A2 Amacrine Gly Release", Gly_a2, "Gly (a.u.)", :red),
]

off_axes = Axis[]
for (i, (name, y, ylabel_txt, color)) in enumerate(off_series)
    ax = Axis(
        fig_off[i, 1],
        title=name,
        xlabel=i == length(off_series) ? "Time (ms)" : "",
        ylabel=ylabel_txt,
    )
    lines!(ax, t_series, y, color=color, linewidth=2)
    vlines!(ax, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
    i < length(off_series) && hidexdecorations!(ax, grid=false)
    push!(off_axes, ax)
end

linkxaxes!(off_axes...)
save("examples/plots/off_pathway_overview.png", fig_off)

#%% Plot all voltages in a single figure
fig4 = Figure(size=(900, 1300))
voltage_series = [
    ("Photoreceptor", V_photo, :blue),
    ("ON Bipolar", V_onbc, :green),
    ("Ganglion", V_gc, :red),
    ("Muller Glia", V_muller, :brown),
]

axes = Axis[]
for (i, (name, v, color)) in enumerate(voltage_series)
    ax = Axis(
        fig4[i, 1],
        xlabel=i == length(voltage_series) ? "Time (ms)" : "",
        ylabel="V (mV)",
        title=name,
    )
    lines!(ax, t_series, v, color=color, linewidth=2)
    vlines!(ax, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
    i < length(voltage_series) && hidexdecorations!(ax, grid=false)
    push!(axes, ax)
end

linkxaxes!(axes...)

save("examples/plots/all_voltages_columns.png", fig4)




