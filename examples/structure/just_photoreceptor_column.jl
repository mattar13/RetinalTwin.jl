# ========================================================================
# just_photoreceptor_column.jl
#
# Assembles a photoreceptor + horizontal cell column, assigns spatial
# coordinates to each population, and saves the mapping here in
# examples/structure/.
#
# Run this once to build the structure. Run scripts load the saved JSON.
# ========================================================================
using RetinalTwin

# ── 1. Build structure ────────────────────────────────────────────────────
#
# PC + HC only. All default build_column connectivity applies:
#   HC ← PC   (Glu, all-to-all)
#   HC ↔ HC   (V,   gap junctions)

model, u0 = build_photoreceptor_column(16; nHC=4)

println("Cells:  ", ordered_cells(model))
println("States: ", length(u0))

# ── 2. Align cell populations ─────────────────────────────────────────────
#
# Every population must have coordinates before distance-based pooling.

align_grid!(model, :PC)                 # 16 PCs → 4×4 grid, coords (1,1)…(4,4)
align_circle!(model, :HC; radius=3.0)   # 4 HCs  → ring centred at origin

# ── 3. Pool connections ───────────────────────────────────────────────────
#
# HC←PC connectivity is all-to-all by default (no rewiring needed here).
# Uncomment below to rewire by distance instead:
#
# pc_syms = [Symbol(:PC, i) for i in 1:16]
# pool_connections!(model, pc_syms, :HC; mode=:nearest_neighbors, k=4)

# ── 4. Save mapping ───────────────────────────────────────────────────────

save_path = joinpath(@__DIR__, "photoreceptor_column.json")
save_mapping(save_path, model, u0)
println("Saved → ", save_path)
