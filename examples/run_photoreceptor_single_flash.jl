# ============================================================
# run_photoreceptor_single_flash.jl — Standalone rod model
#
# Single photoreceptor (rod) driven by a flash stimulus.
# Uses only photoreceptor params, ICs, and model.
# ============================================================
using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie

# ── 1. Parameters and stimulus ───────────────────────────────

rod_params = default_rod_params()

intensity = 1.0   # photons/µm²/ms (small flash)
duration  = 5.0   # ms
t_on      = 5.0  # ms

stim = flash_stimulus(intensity=intensity, duration=duration, t_on=t_on, background=0.0)

println("Standalone rod photoreceptor")
println("  Stimulus: $(intensity) ph/µm²/ms, onset=$(t_on) ms, duration=$(duration) ms")
println("  Phi at t=0: $(compute_stimulus(stim, 0.0))")
println("  Phi at t=$(t_on): $(compute_stimulus(stim, t_on))")
println("  Phi at t=$(t_on + duration + 1.0): $(compute_stimulus(stim, t_on + duration + 1.0))")

# ── 2. Dark-adapted initial conditions (rod only) ────────────

u0 = rod_dark_state(rod_params)
println("Initial rod state (dark adapted): V=$(u0[RetinalTwin.ROD_V_INDEX]) mV, cGMP=$(u0[RetinalTwin.ROD_CGMP_INDEX])")

# ── 3. Rod-only ODE ──────────────────────────────────────────

p = (rod_params, stim, 0.0)
tspan = (0.0, 1200.0)
prob = ODEProblem(rod_rhs!, u0, tspan, p)

println("\nSolving rod-only model...")
sol = solve(prob, Rodas5(); saveat=0.1, abstol=1e-6, reltol=1e-4)

println("  Solver return code: $(sol.retcode)")
println("  Timepoints saved:  $(length(sol.t))")
println("  V range:           $(round(minimum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) mV")
println("  cGMP range:        $(round(minimum(u[RetinalTwin.ROD_CGMP_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_CGMP_INDEX] for u in sol.u), digits=3))")

println("\nDone. Use `sol` for analysis/plotting.")

# -- 4. Plot key rod variables with stimulus span ---------------------------
t = sol.t
vars = [
    ("V (mV)", RetinalTwin.ROD_V_INDEX),
    ("cGMP (uM)", RetinalTwin.ROD_CGMP_INDEX),
    ("Ca_s (uM)", RetinalTwin.ROD_CA_S_INDEX),
    ("Rh* (uM)", RetinalTwin.ROD_RH_INDEX),
    ("PDE (uM)", RetinalTwin.ROD_PDE_INDEX),
    ("Glu (a.u.)", RetinalTwin.ROD_GLU_INDEX),
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
    band!(ax, [t_on, t_on + duration], [y_min, y_min], [y_max, y_max], color=(:gold, 0.2))
    lines!(ax, t, y, color=:black, linewidth=2)
end

display(fig)