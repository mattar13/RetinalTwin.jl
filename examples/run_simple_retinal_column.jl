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
println("    - mGluR6 g_TRPM1 = $(params.ON_BIPOLAR_PARAMS.g_TRPM1) nS")
println("    - mGluR6 tau = $(params.ON_BIPOLAR_PARAMS.tau_mGluR6) ms")

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
stim_params = (
    stim_start = 50.0,    # ms
    stim_end = 55.0,      # ms (5 ms flash)
    photon_flux = 10.0,   # photons/µm²/ms
)

println("\n3. Stimulus configuration:")
println("  ✓ Intensity: $(stim_params.photon_flux) ph/µm²/ms")
println("  ✓ Duration: $(stim_params.stim_start) - $(stim_params.stim_end) ms")

# ── 4. Solve ODE system ───────────────────────────────────────
println("\n4. Solving coupled system...")

tspan = (0.0, 300.0)  # ms
prob = ODEProblem(retinal_column_model!, u0, tspan, (params, stim_params))

sol = solve(prob, Rodas5(); saveat=1.0, abstol=1e-6, reltol=1e-4)

println("  ✓ Solver status: $(sol.retcode)")
println("  ✓ Time points: $(length(sol.t))")

# ── 5. Extract results ────────────────────────────────────────
println("\n5. Extracting results...")

t = sol.t

# Photoreceptor variables
V_photo = [u[RetinalTwin.ROD_V_INDEX] for u in sol.u]
G_photo = [u[RetinalTwin.ROD_G_INDEX] for u in sol.u]
Ca_photo = [u[RetinalTwin.ROD_CA_S_INDEX] for u in sol.u]
Glu_photo = [u[RetinalTwin.ROD_GLU_INDEX] for u in sol.u]  # Glutamate is now a state variable!

# ON bipolar variables
V_onbc = [u[22] for u in sol.u]  # V
w_onbc = [u[23] for u in sol.u]  # w
S_mGluR6 = [u[24] for u in sol.u]  # mGluR6 cascade state
Glu_onbc = [u[25] for u in sol.u]  # ON BC glutamate release

println("  ✓ Photoreceptor V: $(round(minimum(V_photo), digits=2)) → $(round(maximum(V_photo), digits=2)) mV")
println("  ✓ Photoreceptor Glu: $(round(minimum(Glu_photo), digits=3)) → $(round(maximum(Glu_photo), digits=3)) µM")
println("  ✓ ON bipolar V: $(round(minimum(V_onbc), digits=2)) → $(round(maximum(V_onbc), digits=2)) mV")
println("  ✓ mGluR6 state S: $(round(minimum(S_mGluR6), digits=3)) → $(round(maximum(S_mGluR6), digits=3))")

# ── 6. Visualization ──────────────────────────────────────────
println("\n6. Creating plots...")

fig = Figure(size=(1200, 900))

# Photoreceptor voltage
ax1 = Axis(fig[1, 1:2],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="Photoreceptor Membrane Potential")
lines!(ax1, t, V_photo, color=:blue, linewidth=2, label="Photoreceptor")
vlines!(ax1, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax1, position=:rt)

# Photoreceptor cGMP
ax2 = Axis(fig[2, 1],
           xlabel="Time (ms)",
           ylabel="cGMP (µM)",
           title="Photoreceptor cGMP")
lines!(ax2, t, G_photo, color=:darkblue, linewidth=2)
vlines!(ax2, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

# Photoreceptor calcium
ax3 = Axis(fig[2, 2],
           xlabel="Time (ms)",
           ylabel="Ca²⁺ (µM)",
           title="Photoreceptor Calcium")
lines!(ax3, t, Ca_photo, color=:purple, linewidth=2)
vlines!(ax3, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)

# Glutamate release
ax4 = Axis(fig[3, 1:2],
           xlabel="Time (ms)",
           ylabel="Glutamate (µM)",
           title="Glutamate Release (Photoreceptor → ON Bipolar)")
lines!(ax4, t, Glu_photo, color=:orange, linewidth=2, label="Glu released")
vlines!(ax4, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax4, position=:rt)

# mGluR6 cascade state
ax5 = Axis(fig[4, 1],
           xlabel="Time (ms)",
           ylabel="S (cascade state)",
           title="ON Bipolar mGluR6 State")
lines!(ax5, t, S_mGluR6, color=:darkgreen, linewidth=2)
vlines!(ax5, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)
hlines!(ax5, [0.0], color=:black, linestyle=:dot, alpha=0.3)

# ON bipolar voltage
ax6 = Axis(fig[4, 2],
           xlabel="Time (ms)",
           ylabel="V (mV)",
           title="ON Bipolar Membrane Potential")
lines!(ax6, t, V_onbc, color=:green, linewidth=2, label="ON Bipolar")
vlines!(ax6, [stim_params.stim_start, stim_params.stim_end],
        color=:gray, linestyle=:dash, alpha=0.5)
axislegend(ax6, position=:rt)

# Link x-axes
linkxaxes!(ax1, ax2, ax3, ax4, ax5, ax6)

display(fig)