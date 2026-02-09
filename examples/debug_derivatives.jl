# ============================================================
# debug_derivatives.jl — Diagnose which cell types have unstable derivatives
# ============================================================

using RetinalTwin

# Build system
col = build_retinal_column(regime=:scotopic)
sidx = StateIndex(col)
conns = default_connections()
u0 = dark_adapted_state(col, sidx)
p = (col, sidx, conns)

# Compute derivatives at t=0
du = similar(u0)
retinal_column_rhs!(du, u0, p, 0.0)

println("Derivative analysis at t=0 (dark-adapted state):")
println("=" ^ 60)

# Rod 1
idx = sidx.rod[1]
println("\nRod 1 (indices $(idx):$(idx+ROD_STATE_VARS-1)):")
println("  dV/dt        = $(du[idx])  [V = $(u0[idx]) mV]")
println("  dmKv/dt      = $(du[idx+1])")
println("  dhKv/dt      = $(du[idx+2])")
println("  dCa_s/dt     = $(du[idx+5])")
println("  dcGMP/dt     = $(du[idx+17])")
println("  dGlu/dt      = $(du[idx+18])")

# ON-BC
idx = sidx.on_bc[1]
println("\nON-Bipolar (indices $(idx):$(idx+3)):")
println("  dV/dt          = $(du[idx])  [V = $(u0[idx]) mV]")
println("  dw/dt          = $(du[idx+1])")
println("  dS_mGluR6/dt   = $(du[idx+2])  [S = $(u0[idx+2])]")
println("  dGlu/dt        = $(du[idx+3])")

# A2 amacrine
idx = sidx.a2[1]
println("\nA2 Amacrine 1 (indices $(idx):$(idx+2)):")
println("  dV/dt   = $(du[idx])  [V = $(u0[idx]) mV]")
println("  dw/dt   = $(du[idx+1])")
println("  dGly/dt = $(du[idx+2])")

# GABA amacrine
idx = sidx.gaba_ac[1]
println("\nGABA Amacrine 1 (indices $(idx):$(idx+2)):")
println("  dV/dt    = $(du[idx])  [V = $(u0[idx]) mV]")
println("  dw/dt    = $(du[idx+1])")
println("  dGABA/dt = $(du[idx+2])")

# Ganglion
idx = sidx.gc[1]
println("\nGanglion cell (indices $(idx):$(idx+1)):")
println("  dV/dt = $(du[idx])  [V = $(u0[idx]) mV]")
println("  dw/dt = $(du[idx+1])")

# Müller
idx = sidx.muller[1]
println("\nMüller glia (indices $(idx):$(idx+3)):")
println("  dV_M/dt       = $(du[idx])  [V_M = $(u0[idx]) mV]")
println("  dK_o_end/dt   = $(du[idx+1])")
println("  dK_o_stalk/dt = $(du[idx+2])")
println("  dGlu_o/dt     = $(du[idx+3])")

# RPE
idx = sidx.rpe[1]
println("\nRPE (indices $(idx):$(idx+1)):")
println("  dV_RPE/dt = $(du[idx])  [V_RPE = $(u0[idx]) mV]")
println("  dK_sub/dt = $(du[idx+1])")

# Summary stats
println("\n" * "=" ^ 60)
println("Overall derivative statistics:")
println("  Max |du|:  $(maximum(abs, du))")
println("  Min du:    $(minimum(du))")
println("  Max du:    $(maximum(du))")
println("  All finite? $(all(isfinite, du))")

# Identify fastest derivatives
top_idx = sortperm(abs.(du), rev=true)[1:10]
println("\nTop 10 fastest-changing variables (by |du|):")
for (rank, i) in enumerate(top_idx)
    # Figure out which cell type
    cell_type = if i in sidx.rod
        "Rod $(div(i - sidx.rod[1], ROD_STATE_VARS) + 1), var $(mod(i - sidx.rod[1], ROD_STATE_VARS) + 1)"
    elseif i in sidx.cone
        "Cone $(div(i - sidx.cone[1], CONE_STATE_VARS) + 1), var $(mod(i - sidx.cone[1], CONE_STATE_VARS) + 1)"
    elseif i in sidx.hc
        "HC $(div(i - sidx.hc[1], 3) + 1), var $(mod(i - sidx.hc[1], 3) + 1)"
    elseif i in sidx.on_bc
        "ON-BC, var $(i - sidx.on_bc[1] + 1)"
    elseif i in sidx.off_bc
        "OFF-BC, var $(i - sidx.off_bc[1] + 1)"
    elseif i in sidx.a2
        "A2 $(div(i - sidx.a2[1], 3) + 1), var $(mod(i - sidx.a2[1], 3) + 1)"
    elseif i in sidx.gaba_ac
        "GABA $(div(i - sidx.gaba_ac[1], 3) + 1), var $(mod(i - sidx.gaba_ac[1], 3) + 1)"
    elseif i in sidx.da_ac
        "DA, var $(i - sidx.da_ac[1] + 1)"
    elseif i in sidx.gc
        "GC, var $(i - sidx.gc[1] + 1)"
    elseif i in sidx.muller
        "Müller, var $(i - sidx.muller[1] + 1)"
    elseif i in sidx.rpe
        "RPE, var $(i - sidx.rpe[1] + 1)"
    else
        "Unknown index $i"
    end
    println("  $(rank). Index $i ($cell_type): du = $(du[i]), u = $(u0[i])")
end
