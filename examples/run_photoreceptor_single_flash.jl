# ============================================================
# run_photoreceptor_single_flash.jl — Standalone rod model
#
# Single photoreceptor (rod) driven by a flash stimulus.
# Uses only photoreceptor params, ICs, and model.
# ============================================================
using RetinalTwin
using DifferentialEquations

# ── 1. Parameters and stimulus ───────────────────────────────

rod_params = default_rod_params()

intensity = 1.0   # photons/µm²/ms (small flash)
duration  = 5.0   # ms
t_on      = 50.0  # ms

stim = flash_stimulus(intensity=intensity, duration=duration, t_on=t_on, background=0.0)

println("Standalone rod photoreceptor")
println("  Stimulus: $(intensity) ph/µm²/ms, onset=$(t_on) ms, duration=$(duration) ms")
println("  Phi at t=0: $(compute_stimulus(stim, 0.0))")
println("  Phi at t=$(t_on): $(compute_stimulus(stim, t_on))")
println("  Phi at t=$(t_on + duration + 1.0): $(compute_stimulus(stim, t_on + duration + 1.0))")

# ── 2. Dark-adapted initial conditions (rod only) ────────────

function rod_dark_state(params::RodPhotoreceptorParams)
    u0 = zeros(RetinalTwin.ROD_STATE_VARS)
    u0[RetinalTwin.ROD_V_INDEX] = -36.186
    u0[RetinalTwin.ROD_MKV_INDEX] = 0.430
    u0[RetinalTwin.ROD_HKV_INDEX] = 0.999
    u0[RetinalTwin.ROD_MCA_INDEX] = 0.436
    u0[RetinalTwin.ROD_MKCA_INDEX] = 0.642
    u0[RetinalTwin.ROD_CA_S_INDEX] = 0.0966
    u0[RetinalTwin.ROD_CA_F_INDEX] = 0.0966
    u0[RetinalTwin.ROD_CAB_LS_INDEX] = 80.929
    u0[RetinalTwin.ROD_CAB_HS_INDEX] = 29.068
    u0[RetinalTwin.ROD_CAB_LF_INDEX] = 80.929
    u0[RetinalTwin.ROD_CAB_HF_INDEX] = 29.068
    u0[RetinalTwin.ROD_RH_INDEX] = 0.0
    u0[RetinalTwin.ROD_RHI_INDEX] = 0.0
    u0[RetinalTwin.ROD_TR_INDEX] = 0.0
    u0[RetinalTwin.ROD_PDE_INDEX] = 0.0
    u0[RetinalTwin.ROD_CA_PHOTO_INDEX] = 0.3
    u0[RetinalTwin.ROD_CAB_PHOTO_INDEX] = 34.88
    u0[RetinalTwin.ROD_CGMP_INDEX] = 2.0
    u0[RetinalTwin.ROD_GLU_INDEX] = RetinalTwin.rod_glutamate_release(u0[RetinalTwin.ROD_V_INDEX], params)
    return u0
end

u0 = rod_dark_state(rod_params)
println("Initial rod state (dark adapted): V=$(u0[RetinalTwin.ROD_V_INDEX]) mV, cGMP=$(u0[RetinalTwin.ROD_CGMP_INDEX])")

# ── 3. Rod-only ODE ──────────────────────────────────────────

function rod_rhs!(du, u, p, t)
    params, stim = p
    Phi = compute_stimulus(stim, t)
    rod_cell_rhs!(du, u, (params, Phi, 0.0), t)
    return nothing
end

p = (rod_params, stim)
tspan = (0.0, 200.0)
prob = ODEProblem(rod_rhs!, u0, tspan, p)

println("\nSolving rod-only model...")
sol = solve(prob, Rodas5(); saveat=0.1, abstol=1e-6, reltol=1e-4)

println("  Solver return code: $(sol.retcode)")
println("  Timepoints saved:  $(length(sol.t))")
println("  V range:           $(round(minimum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_V_INDEX] for u in sol.u), digits=3)) mV")
println("  cGMP range:        $(round(minimum(u[RetinalTwin.ROD_CGMP_INDEX] for u in sol.u), digits=3)) → $(round(maximum(u[RetinalTwin.ROD_CGMP_INDEX] for u in sol.u), digits=3))")

println("\nDone. Use `sol` for analysis/plotting.")
