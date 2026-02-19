using Revise
using RetinalTwin
using DifferentialEquations

include("retinal_column_plotting.jl")

println("=" ^ 60)
println("RetinalColumnModel Example (Full Build)")
println("=" ^ 60)

#%% --------- Build the model ---------
params = default_retinal_params()
pc_coords = square_grid_coords(16)
model, u0 = build_column(pc_coords=pc_coords)

println("Cells: ", ordered_cells(model))
println("Total states: ", length(u0))

# Dark adaptation
tspan_dark = (0.0, 2000.0)
stim_dark = RetinalTwin.make_uniform_flash_stimulus(photon_flux=0.0)
prob_dark = ODEProblem(model, u0, tspan_dark, (params, stim_dark))
sol_dark = solve(prob_dark, Rodas5(); save_everystep=false, save_start=false, save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol_dark.u[end]

# Flash run
stim_start = 80.0
stim_end = 180.0
photon_flux = 0.10

# Exponential spot flash centered on the PC grid
x_center = sum(first.(pc_coords)) / length(pc_coords)/4
y_center = sum(last.(pc_coords)) / length(pc_coords)
selected_stimulus = RetinalTwin.make_exponential_spot_stimulus(
    stim_start=stim_start,
    stim_end=stim_end,
    photon_flux=photon_flux,
    center_x=x_center,
    center_y=y_center,
    decay_length=0.5,
)

fig = plot_cell_layout_3d(
    model;
    z_stim=0.0,
    stimulus_func=selected_stimulus,
    stimulus_time=(stim_start + stim_end) / 2,
    stimulus_grid_n=61,
    savepath=joinpath(@__DIR__, "plots", "column_layout_3d.png"),
)
display(fig)


#%% --------- Run the model ---------
tspan = (0.0, 1200.0)
prob = ODEProblem(model, u0, tspan, (params, selected_stimulus))
@time sol = solve(prob, Rodas5(); tstops = [stim_start, stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)

println("Solver status: ", sol.retcode)
println("Saved time points: ", length(sol.t))


fig = plot_all_cells_v_ca_release(
    sol,
    model;
    stim_start=stim_start,
    stim_end=stim_end,
    savepath=joinpath(@__DIR__, "plots", "retinal_column_all_cells.png"),
)