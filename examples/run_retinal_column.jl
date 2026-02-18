using RetinalTwin
using DifferentialEquations

include("retinal_column_plotting.jl")

println("=" ^ 60)
println("RetinalColumnModel Example (Full Build)")
println("=" ^ 60)

params = default_retinal_params()
model, u0 = build_column()

println("Cells: ", ordered_cells(model))
println("Total states: ", length(u0))

# Dark adaptation
tspan_dark = (0.0, 2000.0)
stim_dark(t) = RetinalTwin.single_flash(t; photon_flux=0.0)
prob_dark = ODEProblem(model, u0, tspan_dark, (params, stim_dark))
sol_dark = solve(prob_dark, Rodas5(); save_everystep=false, save_start=false, save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol_dark.u[end]

# Flash run
stim_start = 80.0
stim_end = 180.0
photon_flux = 0.001
stim_func(t) = RetinalTwin.single_flash(t; stim_start=stim_start, stim_end=stim_end, photon_flux=photon_flux)

tspan = (0.0, 1200.0)
prob = ODEProblem(model, u0, tspan, (params, stim_func))
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
