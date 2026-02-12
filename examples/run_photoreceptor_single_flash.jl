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

photoreceptor_params = default_rod_params()

println("Standalone rod photoreceptor")
println("  Stimulus: $(stim_params.photon_flux) ph/µm²/ms, onset=$(stim_params.stim_start) ms")

# ── 2. Dark-adapted initial conditions ────────────────────────

u0 = rod_dark_state(photoreceptor_params)
println("Initial photoreceptor state (dark adapted): V=$(u0[RetinalTwin.ROD_V_INDEX]) mV, G=$(u0[RetinalTwin.ROD_G_INDEX])")

# ── 3. Rod ODE ─────────────────────────────────────────────────

println("\n1. Computing steady state (dark, no stimulus)...")
stim_params_dark = (
    stim_start = 0.0,
    stim_end = 0.0,
    photon_flux = 0.0
)

tspan = (0.0, 1000.0)
prob = SteadyStateProblem(retinal_column_model!, u0, (params, stim_params_dark))
ss_sol = solve(prob, DynamicSS(Rodas5()); abstol=1e-6, reltol=1e-4)
u0 = ss_sol.u


# ── 4. Stimulus parameters ────────────────────────────────────
stim_params = (
    stim_start = 50.0,    # ms
    stim_end = 150.0,      # ms (5 ms flash)
    photon_flux = 100.0,   # photons/µm²/ms
)

p = (photoreceptor_params, stim_params)
tspan = (0.0, 500.0)
prob = ODEProblem(photoreceptor_model!, u0, tspan, p)

println("\nSolving photoreceptor model...")
sol = solve(prob, Rodas5(); tstops = [stim_params.stim_start, stim_params.stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)

println("  Solver return code: $(sol.retcode)")
println("  Timepoints saved:  $(length(sol.t))")    
println("  V range:           $(round(minimum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) mV")
println("  G range:           $(round(minimum(u[RetinalTwin.ROD_G_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_G_INDEX] for u in sol.u), digits=3))")


# ── 4. Plot key rod variables ─────────────────────────────────
t = sol.t
vars = [
    ("V (mV)", RetinalTwin.ROD_V_INDEX),
    ("Glu (µM)", RetinalTwin.ROD_GLU_INDEX),
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
    band!(ax, [stim_params.stim_start, stim_params.stim_end], [y_min, y_min], [y_max, y_max], color=(:gold, 0.2))
    lines!(ax, t, y, color=:black, linewidth=2)
end

display(fig)
