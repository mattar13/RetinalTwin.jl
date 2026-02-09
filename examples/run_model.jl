# ============================================================
# run_model.jl — Step-by-step retinal column simulation
#
# Everything is explicit here. No hidden helper functions.
# Run section by section in the REPL to see exactly what's happening.
# ============================================================
using Revise
using RetinalTwin
#If we have revised the code, 
using DifferentialEquations
using Setfield

# ── 1. Build the retinal column ──────────────────────────────

col = build_retinal_column(regime=:scotopic)
sidx = StateIndex(col)

println("Retinal column: $(sidx.total) ODEs")
println("  Rods: $(col.pop.n_rod)  |  Cones: $(col.pop.n_cone)")
println("  HC: $(col.pop.n_hc)  |  ON-BC: $(col.pop.n_on)  |  OFF-BC: $(col.pop.n_off)")
println("  A2: $(col.pop.n_a2)  |  GABA: $(col.pop.n_gaba)  |  DA: $(col.pop.n_dopa)")
println("  GC: $(col.pop.n_gc)  |  Müller: $(col.pop.n_muller)  |  RPE: $(col.pop.n_rpe)")

# State vector layout (flat vector, 1-indexed ranges):
println("\nState index ranges:")
println("  rod:     $(sidx.rod)      (6 vars × $(col.pop.n_rod): R*, G, Ca, V, h, Glu)")
println("  cone:    $(sidx.cone)  (6 vars × $(col.pop.n_cone): R*, G, Ca, V, h, Glu)")
println("  hc:      $(sidx.hc)  (3 vars × $(col.pop.n_hc): V, w, s_Glu)")
println("  on_bc:   $(sidx.on_bc)  (4 vars × $(col.pop.n_on): V, w, S_mGluR6, Glu)")
println("  off_bc:  $(sidx.off_bc)  (4 vars × $(col.pop.n_off): V, w, s_Glu, Glu)")
println("  a2:      $(sidx.a2)  (3 vars × $(col.pop.n_a2): V, w, Gly)")
println("  gaba_ac: $(sidx.gaba_ac)  (3 vars × $(col.pop.n_gaba): V, w, GABA)")
println("  da_ac:   $(sidx.da_ac)  (3 vars × $(col.pop.n_dopa): V, w, DA)")
println("  gc:      $(sidx.gc)  (2 vars × $(col.pop.n_gc): V, w)")
println("  muller:  $(sidx.muller)  (4 vars × $(col.pop.n_muller): V_M, K_o_end, K_o_stalk, Glu_o)")
println("  rpe:     $(sidx.rpe)  (2 vars × $(col.pop.n_rpe): V_RPE, K_sub)")

# ── 2. Configure stimulus ────────────────────────────────────

intensity = 10.0      # photons/µm²/ms (dim scotopic flash)
duration  = 10.0      # ms
t_on      = 200.0     # ms — flash onset (give 200 ms dark baseline)

col = @set col.stimulus.I_0    = intensity
col = @set col.stimulus.t_dur  = duration
col = @set col.stimulus.t_on   = t_on

println("\nStimulus: $(intensity) ph/µm²/ms, onset=$(t_on) ms, duration=$(duration) ms")
println("  Background: $(col.stimulus.background)")

# Verify the stimulus function works:
println("  Phi at t=0:   $(compute_stimulus(col.stimulus, 0.0))")
println("  Phi at t=205: $(compute_stimulus(col.stimulus, 205.0))")
println("  Phi at t=220: $(compute_stimulus(col.stimulus, 220.0))")

# ── 3. Connection table ──────────────────────────────────────

conns = default_connections()
println("\n$(length(conns)) synaptic connections:")
for c in conns
    println("  $(c.pre) → $(c.post)  [$(c.nt_type), $(c.receptor), g=$(c.g_max) nS, E=$(c.E_rev) mV]")
end

# ── 4. Initial conditions (dark-adapted) ─────────────────────

u0 = dark_adapted_state(col, sidx)
println("\nInitial state vector: $(length(u0)) elements")

# Spot-check a few values:
rod1_offset = sidx.rod[1]
println("  Rod 1: R*=$(u0[rod1_offset]), G=$(u0[rod1_offset+1]), Ca=$(u0[rod1_offset+2]), V=$(u0[rod1_offset+3]) mV, h=$(u0[rod1_offset+4]), Glu=$(u0[rod1_offset+5])")

on_offset = sidx.on_bc[1]
println("  ON-BC: V=$(u0[on_offset]) mV, w=$(u0[on_offset+1]), S_mGluR6=$(u0[on_offset+2]), Glu=$(u0[on_offset+3])")

gc_offset = sidx.gc[1]
println("  GC:    V=$(u0[gc_offset]) mV, w=$(u0[gc_offset+1])")

println("  All finite? $(all(isfinite, u0))")
println("  Any NaN?    $(any(isnan, u0))")

# ── 5. Set up and solve the ODE ──────────────────────────────

t_total  = 1000.0   # ms simulation window
dt_save  = 0.1      # ms save interval
tspan    = (0.0, t_total)
p        = (col, sidx, conns)

println("\nSetting up ODE problem...")
println("  tspan: $(tspan)")
println("  dt_save: $(dt_save) ms → expect ~$(Int(t_total / dt_save)) saved timepoints")

prob = ODEProblem(retinal_column_rhs!, u0, tspan, p)

# Test the RHS at t=0 to make sure it doesn't error or produce NaN:
du_test = similar(u0)
retinal_column_rhs!(du_test, u0, p, 0.0)
println("  RHS at t=0: all finite? $(all(isfinite, du_test))")
println("  RHS at t=0: any NaN?    $(any(isnan, du_test))")
println("  RHS max |du|: $(maximum(abs, du_test))")

# Also test at flash onset:
retinal_column_rhs!(du_test, u0, p, t_on + 1.0)
println("  RHS at t=$(t_on+1): all finite? $(all(isfinite, du_test))")
println("  RHS at t=$(t_on+1): max |du|: $(maximum(abs, du_test))")

println("\nSolving ODE (Rodas5 — stiff implicit solver)...")
sol = solve(prob, Rodas5();
            saveat=dt_save,
            abstol=1e-6, reltol=1e-4,
            maxiters=1_000_000)

println("  Solver return code: $(sol.retcode)")
println("  Timepoints saved:  $(length(sol.t))")
println("  Time range:        $(sol.t[1]) → $(sol.t[end]) ms")
println("  Solution finite?   $(all(u -> all(isfinite, u), sol.u))")

if length(sol.t) < 10
    println("\n⚠ WARNING: Very few timepoints returned. The solver may have failed.")
    println("  sol.t = $(sol.t)")
    if length(sol.u) >= 1
        u_last = sol.u[end]
        println("  Last state has NaN? $(any(isnan, u_last))")
        println("  Last state has Inf? $(any(isinf, u_last))")
        # Find which variables went bad:
        bad_idx = findall(x -> !isfinite(x), u_last)
        if !isempty(bad_idx)
            println("  Non-finite at indices: $(bad_idx)")
        end
    end
    println("\nStopping here — fix the solver issue before continuing.")
else
    # ── 6. Inspect raw solution ──────────────────────────────────

    println("\nSpot-checking solution at key timepoints:")
    for t_check in [0.0, 100.0, t_on, t_on + 5.0, t_on + 50.0, t_on + 200.0, 500.0]
        ti = argmin(abs.(sol.t .- t_check))
        u = sol.u[ti]
        rod_V = u[rod1_offset + 3]
        on_V  = u[on_offset]
        gc_V  = u[gc_offset]
        println("  t=$(round(sol.t[ti], digits=1)) ms:  Rod V=$(round(rod_V, digits=2)) mV,  ON-BC V=$(round(on_V, digits=2)) mV,  GC V=$(round(gc_V, digits=2)) mV")
    end

    # ── 7. Compute ERG ───────────────────────────────────────────

    println("\nComputing ERG field potential...")
    erg, components = compute_erg(sol, col, sidx)
    println("  ERG trace length: $(length(erg))")
    println("  Components: $(keys(components))")
    println("  a-wave min:  $(round(minimum(components[:a_wave]), digits=4))")
    println("  b-wave max:  $(round(maximum(components[:b_wave]), digits=4))")
    println("  OP range:    $(round(minimum(components[:OPs]), digits=4)) to $(round(maximum(components[:OPs]), digits=4))")
    println("  c-wave max:  $(round(maximum(components[:c_wave]), digits=4))")

    # ── 8. Extract voltages ──────────────────────────────────────

    println("\nExtracting cell voltages...")
    voltages = extract_voltages(sol, col, sidx)
    for (name, V) in voltages
        println("  $(name): $(size(V, 2)) cells, V range = $(round(minimum(V), digits=1)) to $(round(maximum(V), digits=1)) mV")
    end

    # ── 9. Build result tuple (for plotting functions) ───────────

    result = (t=sol.t, erg=erg, erg_components=components,
              cell_voltages=voltages, solution=sol, col=col, sidx=sidx)

    # ── 10. Plot with GLMakie ────────────────────────────────────
    show_plots = true
    if show_plots
        fig_erg = plot_erg_gl(result, show_components=true)
        display(fig_erg)

        fig_v = plot_cell_voltages_gl(result,
            cell_types=[:rod, :cone, :on_bc, :off_bc, :a2, :gaba_ac, :gc])
        display(fig_v)

        fig_photo = plot_phototransduction_breakdown(result, cell=:rod, cell_index=1)
        display(fig_photo)
    end

    println("\nDone. Result tuple available as `result`.")
    println("GLMakie plots displayed: ERG, cell voltages, phototransduction breakdown")
end
