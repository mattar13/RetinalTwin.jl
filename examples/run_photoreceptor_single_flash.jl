# ============================================================
# run_photoreceptor_single_flash.jl — Standalone rod model
#
# Single photoreceptor (rod) driven by a flash stimulus.
# ============================================================
using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie

# ── 1. Parameters and stimulus ───────────────────────────────

rod_params = default_rod_params()

# Stimulus parameters
intensity = 10.0   # photons/µm²/ms (small flash)
stim_start = 50.0  # ms
stim_end = 55.0    # ms (instantaneous flash)
v_hold = false     # don't clamp voltage
I_feedback = 0.0   # pA

println("Standalone rod photoreceptor")
println("  Stimulus: $(intensity) ph/µm²/ms, onset=$(stim_start) ms")

# ── 2. Dark-adapted initial conditions ────────────────────────

u0 = rod_dark_state(rod_params)
println("Initial rod state (dark adapted): V=$(u0[RetinalTwin.ROD_V_INDEX]) mV, G=$(u0[RetinalTwin.ROD_G_INDEX])")

# ── 3. Rod ODE ─────────────────────────────────────────────────

p = (rod_params, stim_start, stim_end, intensity, v_hold, I_feedback)
tspan = (0.0, 500.0)
prob = ODEProblem(rod_model!, u0, tspan, p)

println("\nSolving rod-only model...")
sol = solve(prob, Rodas5(); saveat=1.0, abstol=1e-6, reltol=1e-4)

println("  Solver return code: $(sol.retcode)")
println("  Timepoints saved:  $(length(sol.t))")
println("  V range:           $(round(minimum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) mV")
println("  G range:           $(round(minimum(u[RetinalTwin.ROD_G_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_G_INDEX] for u in sol.u), digits=3))")


# ── 4. Plot key rod variables ─────────────────────────────────
t = sol.t
vars = [
    ("V (mV)", RetinalTwin.ROD_V_INDEX),
    ("G/cGMP (µM)", RetinalTwin.ROD_G_INDEX),
    ("Ca_s (µM)", RetinalTwin.ROD_CA_S_INDEX),
    ("R* (µM)", RetinalTwin.ROD_R_INDEX),
    ("P* (µM)", RetinalTwin.ROD_P_INDEX),
]

fig = Figure(size=(900, 220 * length(vars)))

for (i, (label, idx)) in enumerate(vars)
    y = getindex.(sol.u, idx)
    y_min = minimum(y)
    y_max = maximum(y)
    if y_min == y_max
        y_min -= 1.0
        y_max += 1.0
    end

    ax = Axis(fig[i, 1], title=label, xlabel="Time (ms)", ylabel=label)
    band!(ax, [stim_start, stim_end], [y_min, y_min], [y_max, y_max], color=(:gold, 0.2))
    lines!(ax, t, y, color=:black, linewidth=2)
end

display(fig)
