using RetinalTwin
using DifferentialEquations

include("retinal_column_plotting.jl")

println("=" ^ 60)
println("RetinalColumnModel Example (Full Build)")
println("=" ^ 60)

params = default_retinal_params()
pc_coords = square_grid_coords(16)
model, u0 = build_column(nPC=16, nONBC=2, pc_coords=pc_coords)

println("Cells: ", ordered_cells(model))
println("Total states: ", length(u0))

# Dark adaptation
tspan_dark = (0.0, 2000.0)
stim_dark(t, x, y) = RetinalTwin.uniform_flash(t, x, y; photon_flux=0.0)
prob_dark = ODEProblem(model, u0, tspan_dark, (params, stim_dark))
sol_dark = solve(prob_dark, Rodas5(); save_everystep=false, save_start=false, save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol_dark.u[end]

# Flash run
stim_start = 80.0
stim_end = 180.0
photon_flux = 10.0

# Spatial flash (subset of PCs by geometry)
selected_stimulus(t, x, y) = RetinalTwin.spatial_stimulus(t, x, y;
    stim_start=stim_start, stim_end=stim_end,
    photon_flux=photon_flux,
    xmin=1.0, xmax=4.0,
    ymin=1.0, ymax=1.0,
)

tspan = (0.0, 1200.0)
prob = ODEProblem(model, u0, tspan, (params, selected_stimulus))
sol = solve(prob, Rodas5(); tstops = [stim_start, stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)

println("Solver status: ", sol.retcode)
println("Saved time points: ", length(sol.t))

plot_all_cells_v_ca_release(
    sol,
    model;
    stim_start=stim_start,
    stim_end=stim_end,
    savepath=joinpath(@__DIR__, "plots", "run_retinal_column_all_cells.png"),
)

plot_cell_layout_3d(
    model;
    z_pc=1.0,
    z_bc=2.0,
    z_gc=3.0,
    savepath=joinpath(@__DIR__, "plots", "run_retinal_column_layout_3d.png"),
)
