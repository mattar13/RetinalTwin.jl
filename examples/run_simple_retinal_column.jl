using RetinalTwin
using DifferentialEquations

println("=" ^ 60)
println("RetinalColumnModel Example (Simple Build)")
println("=" ^ 60)

params = load_all_params()
model, u0 = build_column(nPC=1, nONBC=1, nOFFBC=0, nA2=0, nGC=0, nMG=0)

println("Cells: ", ordered_cells(model))
println("Total states: ", length(u0))

stim_start = 80.0
stim_end = 180.0
photon_flux = 50.0
stim_func(t) = RetinalTwin.single_flash(t; stim_start=stim_start, stim_end=stim_end, photon_flux=photon_flux)

tspan = (0.0, 300.0)
prob = ODEProblem(model, u0, tspan, (params, stim_func))
sol = solve(prob, Rodas5(); saveat=1.0, abstol=1e-6, reltol=1e-4)

println("Solver status: ", sol.retcode)
println("Saved time points: ", length(sol.t))

plot_all_cells_v_ca_release(
    sol,
    model;
    stim_start=stim_start,
    stim_end=stim_end,
    savepath=joinpath(@__DIR__, "plots", "run_simple_retinal_column_all_cells.png"),
)
