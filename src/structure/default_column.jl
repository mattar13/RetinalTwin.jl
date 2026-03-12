# ========================================================================
# default_column.jl
#
# Assembles the full retinal column (PC + HC + ONBC + OFFBC + A2 + GC + MG),
# assigns spatial coordinates to each population, optionally rewires
# connections by distance, and saves the mapping to data/.
#
# Run this once to build the structure. Run scripts load the saved JSON.
# ========================================================================
using RetinalTwin

# ── 1. Build structure ────────────────────────────────────────────────────
#
# build_column wires ONBC←PC by sequential block pooling (4 PCs per ONBC).
# Connectivity wired at this stage:
#   HC   ← PC    (Glu, all-to-all)
#   HC   ↔ HC    (V,   gap junctions)
#   ONBC ← PC    (Glu, block pooling: ONBC1←PC1–4, ONBC2←PC5–8, …)
#   OFFBC← PC    (Glu, all-to-all)
#   A2   ← ONBC  (Glu + V gap junctions)
#   ONBC ← A2    (V,   gap junctions)
#   GC   ← ONBC  (Glu)
#   GC   ← OFFBC (Glu)
#   GC   ← A2    (Y / glycine)

model, u0 = build_column(
    nPC=16, nHC=4, nONBC=4, onbc_pool_size=4,
    nOFFBC=4, nA2=4, nGC=1, nMG=4,
    pc_coords=square_grid_coords(16),
)

println("Cells:  ", ordered_cells(model))
println("States: ", length(u0))

# ── 2. Align cell populations ─────────────────────────────────────────────
#
# Every population must have coordinates before distance-based pooling.
# Adjust radii to match the spatial scale of your PC grid.
# The default PC grid spans (1,1) → (4,4); choose radii accordingly.

align_grid!(model,   :PC)                   # 4×4 grid, coords (1,1)…(4,4)
align_circle!(model, :HC;    radius=3.0)    # outer ring (receives from all PCs)
align_circle!(model, :ONBC;  radius=2.0)    # inner ring
align_circle!(model, :OFFBC; radius=2.0)    # inner ring (same layer as ONBC)
align_circle!(model, :A2;    radius=1.5)    # innermost (between bipolar and GC)
align_grid!(model,   :GC)                   # single cell → (1,1)
align_circle!(model, :MG;    radius=3.5)    # outermost (spans full column depth)

# ── 3. Pool connections ───────────────────────────────────────────────────
#
# The default ONBC←PC wiring from build_column is sequential block pooling.
# To replace it with distance-based spatial pooling, uncomment one of the
# examples below (requires coordinates to be set first).
#
# Nearest-neighbor pooling (each ONBC connects to its 4 nearest PCs):
# pc_syms = [Symbol(:PC, i) for i in 1:16]
# pool_connections!(model, pc_syms, :ONBC; mode=:nearest_neighbors, k=4)
#
# Distance-inclusive pooling (each ONBC connects to all PCs within radius 1.5):
# pool_connections!(model, pc_syms, :ONBC; mode=:distance_inclusive, k=1.5)
#
# Distance-exclusive pooling (each PC connects to its single nearest ONBC):
# pool_connections!(model, pc_syms, :ONBC; mode=:distance_exclusive, k=2.0)

# ── 4. Save mapping ───────────────────────────────────────────────────────

save_path = normpath(joinpath(@__DIR__, "..", "structure", "default_column.json"))
mkpath(dirname(save_path))
save_mapping(save_path, model, u0)
println("Saved → ", save_path)
