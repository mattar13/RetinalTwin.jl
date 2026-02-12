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

println("  ✓ Photoreceptor parameters: $(length(fieldnames(typeof(params.PHOTORECEPTOR_PARAMS)))) fields")
println("  ✓ ON bipolar parameters: $(length(fieldnames(typeof(params.ON_BIPOLAR_PARAMS)))) fields")

# ── 2. Build initial conditions ──────────────────────────────
println("\n2. Building initial conditions...")
u0 = retinal_column_initial_conditions(params)

println("  ✓ Total state variables: $(length(u0))")
println("    - Photoreceptor states: 1-21 (includes glutamate)")
println("    - ON bipolar states: 22-25")
println("  ✓ Photoreceptor V₀ = $(u0[RetinalTwin.ROD_V_INDEX]) mV")
println("  ✓ Photoreceptor Glu₀ = $(u0[RetinalTwin.ROD_GLU_INDEX]) µM")
println("  ✓ ON bipolar V₀ = $(u0[22]) mV")

# ── 3. Define stimulus ────────────────────────────────────────
println("\n3. Computing steady state (dark, no stimulus)...")
stim_params_dark = (
    stim_start = 0.0,
    stim_end = 0.0,
    photon_flux = 0.0
)

tspan = (0.0, 1000.0)
prob = ODEProblem(retinal_column_model!, u0, tspan, (params, stim_params_dark))
sol = solve(prob, Rodas5(); save_everystep=false, save_start=false,save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol.u[end]

stim_params = (
    stim_start = 50.0,    # ms
    stim_end = 55.0,      # ms (5 ms flash)
    photon_flux = 1000.0,   # photons/µm²/ms
)

println("\n4. Stimulus configuration:")
println("  ✓ Intensity: $(stim_params.photon_flux) ph/µm²/ms")
println("  ✓ Duration: $(stim_params.stim_start) - $(stim_params.stim_end) ms")

# ── 4. Solve ODE system ───────────────────────────────────────
println("\n5. Solving coupled system...")

tspan = (0.0, 300.0)  # ms
prob = ODEProblem(retinal_column_model!, u0, tspan, (params, stim_params))

sol = solve(prob, Rodas5(); tstops = [stim_params.stim_start, stim_params.stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)

println("  ✓ Solver status: $(sol.retcode)")
println("  ✓ Time points: $(length(sol.t))")

# ── 5. Extract results ────────────────────────────────────────
println("\n6. Extracting results...")

t = sol.t

# Photoreceptor variables
V_photo = [u[RetinalTwin.ROD_V_INDEX] for u in sol.u]
Ca_photo = [u[RetinalTwin.ROD_CA_S_INDEX] for u in sol.u]
Glu_photo = [u[RetinalTwin.ROD_GLU_INDEX] for u in sol.u]  # Glutamate is now a state variable!

# ON bipolar variables
V_onbc = [u[22] for u in sol.u]  # V
Ca_onbc = [u[25] for u in sol.u]  # Ca
Glu_onbc = [u[27] for u in sol.u]  # ON BC glutamate release

V_offbc = [u[28] for u in sol.u]  # OFF BC V
Ca_offbc = [u[31] for u in sol.u]  # OFF BC Ca
A_offbc = [u[32] for u in sol.u]  # OFF BC iGluR activation
D_offbc = [u[33] for u in sol.u]  # OFF BC desensitization
Glu_offbc = [u[34] for u in sol.u]  # OFF BC glutamate release

println("  ✓ Photoreceptor V: $(round(minimum(V_photo), digits=2)) → $(round(maximum(V_photo), digits=2)) mV")
println("  ✓ Photoreceptor Glu: $(round(minimum(Glu_photo), digits=3)) → $(round(maximum(Glu_photo), digits=3)) µM")
println("  ✓ ON bipolar V: $(round(minimum(V_onbc), digits=2)) → $(round(maximum(V_onbc), digits=2)) mV")
println("  ✓ mGluR6 state S: $(round(minimum(S_mGluR6), digits=3)) → $(round(maximum(S_mGluR6), digits=3))")

#%% ── 6. Visualization ──────────────────────────────────────────
println("\n7. Creating plots...")

fig1 = Figure(size=(1200, 500))

# Photoreceptor voltage
ax1a = Axis(fig1[1, 1],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="Photoreceptor Membrane Potential")
lines!(ax1a, t, V_photo, color=:blue, linewidth=2, label="Photoreceptor")
vlines!(ax1a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

ax1b = Axis(fig1[1, 2],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="ON Bipolar Membrane Potential")
lines!(ax1b, t, V_onbc, color=:green, linewidth=2)
vlines!(ax1b, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

ax1c = Axis(fig1[1, 3],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="OFF Bipolar Membrane Potential")
lines!(ax1c, t, V_offbc, color=:red, linewidth=2)
vlines!(ax1c, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax2a = Axis(fig1[2, 1],
           xlabel="Time (ms)",
           ylabel="Ca (µM)",
           title="Photoreceptor Calcium")
lines!(ax2a, t, Ca_photo, color=:purple, linewidth=2)
vlines!(ax2a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax2b = Axis(fig1[2, 2],
           xlabel="Time (ms)",
           ylabel="Ca (µM)",
           title="ON Bipolar Calcium")
lines!(ax2b, t, Ca_onbc, color=:purple, linewidth=2)
vlines!(ax2b, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax2c = Axis(fig1[2, 3],
           xlabel="Time (ms)",
           ylabel="Ca (µM)",
           title="OFF Bipolar Calcium")
lines!(ax2c, t, Ca_offbc, color=:purple, linewidth=2)
vlines!(ax2c, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

# Glutamate release
ax3a = Axis(fig1[3, 1],
           xlabel="Time (ms)",
           ylabel="Glutamate (µM)",
           title="Glutamate Release")
lines!(ax3a, t, Glu_photo, color=:orange, linewidth=2)
vlines!(ax3a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

ax3b = Axis(fig1[3, 2],
           xlabel="Time (ms)",
           ylabel="Glutamate (µM)",
           title="Glutamate Release")
lines!(ax3b, t, Glu_onbc, color=:green, linewidth=2)
vlines!(ax3b, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

ax3c = Axis(fig1[3, 3],
           xlabel="Time (ms)",
           ylabel="Glutamate (µM)",
           title="Glutamate Release")
lines!(ax3c, t, Glu_offbc, color=:red, linewidth=2)
vlines!(ax3c, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

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
save("examples/plots/voltage_vs_glutamate_release.png", fig2)


#Plot the desensitization of the OFF bipolar iGluR
fig3 = Figure(size=(500, 500))
ax1a = Axis(fig3[1, 1],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="OFF Bipolar Membrane Potential")
lines!(ax1a, t, V_offbc, color=:black, linewidth=2)
vlines!(ax1a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

ax2a = Axis(fig3[2, 1],
           xlabel="Time (ms)",
           ylabel="Activation",
           title="AMPA/KAR-like activation")
lines!(ax2a, t, A_offbc, color=:green, linewidth=2)
vlines!(ax2a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax3a = Axis(fig3[3, 1],
           xlabel="Time (ms)",
           ylabel="Desensitization",
           title="AMPA/KAR-like desensitization")
lines!(ax3a, t, D_offbc, color=:red, linewidth=2)
vlines!(ax3a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)


ax4a = Axis(fig3[4, 1],
           xlabel="Time (ms)",
           ylabel="Glutamate Release (µM)",
           title="OFF Bipolar Glutamate Release")
lines!(ax4a, t, A_offbc .* D_offbc, color=:orange, linewidth=2)
vlines!(ax4a, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

save("examples/plots/desensitization_off_bipolar_iGluR.png", fig3)
