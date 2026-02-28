# ========================================================================
# retinal_column_plotting.jl
#
# Loads the default retinal column (built by examples/structure/default_column.jl)
# and plots the 3D cell layout, with a uniform flash stimulus plane overlaid.
#
# Prerequisites (run once):
#   julia examples/structure/default_column.jl
# ========================================================================
using RetinalTwin
using CairoMakie

# ── 1. Load pre-built column ─────────────────────────────────────────────

col_path = joinpath(@__DIR__, "structure", "default_column.json")
isfile(col_path) || error("Column map not found — run examples/structure/default_column.jl first.")
model, _u0 = load_mapping(col_path)

println("Cells: ", ordered_cells(model))

# ── 2. Stimulus ───────────────────────────────────────────────────────────

stim_start = 200.0   # ms
stim_end   = 250.0   # ms
stim_amp   = 1.0     # photon flux (arbitrary units)

selected_stimulus = make_uniform_flash_stimulus(;
    stim_start  = stim_start,
    stim_end    = stim_end,
    photon_flux = stim_amp,
)

# ── 3. Plot ───────────────────────────────────────────────────────────────
#
# z coordinates come from erg_depth_map.csv (loaded inside plot_cell_layout_3d).
# x,y come from CellRef coordinates set by align_grid! / align_circle!
# in default_column.jl.  Cells that span a depth range are shown as
# vertical line segments; the scatter marker sits at the z midpoint.

fig = plot_cell_layout_3d(
    model;
    z_stim          = 0.0,
    stimulus_func   = selected_stimulus,
    stimulus_time   = (stim_start + stim_end) / 2,
    stimulus_grid_n = 61,
    savepath        = joinpath(@__DIR__, "plots", "uniform_flash_column_layout_3d.png"),
)

display(fig)
