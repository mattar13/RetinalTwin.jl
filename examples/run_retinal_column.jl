using RetinalTwin
using DifferentialEquations
using CairoMakie

println("=" ^ 60)
println("RetinalColumnModel Example (Phase-1: PC -> ONBC)")
println("=" ^ 60)

# Build parameters and network model
params = default_retinal_params()
model, u0 = build_column(nPC=4, nONBC=1, nGC=1, nA2=0)

println("Cells: ", collect(keys(model.cells)))``
println("Total states: ", length(u0))

# Dark-adapted steady state
tspan = (0.0, 1000.0)
stim_func_dark(t) = RetinalTwin.single_flash(t;photon_flux = 0.0)
prob = ODEProblem(model, u0, tspan, (params, stim_func_dark))
sol = solve(prob, Rodas5(); save_everystep=false, save_start=false,save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol.u[end]

# Flash stimulus
stim_start = 80.0
stim_end = 180.0
photon_flux = 50.0
stim_func(t) = RetinalTwin.single_flash(
    t;
    stim_start=stim_start,
    stim_end=stim_end,
    photon_flux=photon_flux
)

# Solve ODE
tspan = (0.0, 300.0)
prob = ODEProblem(model, u0, tspan, (params, stim_func))
sol = solve(prob, Rodas5(); saveat=1.0, abstol=1e-6, reltol=1e-4)

println("Solver status: ", sol.retcode)
println("Saved time points: ", length(sol.t))

# Report final values from mapped outputs
u_end = sol.u[end]
pc = model.cells[:PC1]
onbc = model.cells[:ONBC1]
println("Final PC V: ", get_out(u_end, pc, :V))
println("Final PC Glu: ", get_out(u_end, pc, :Glu))
println("Final ONBC V: ", get_out(u_end, onbc, :V))
println("Final ONBC Glu: ", get_out(u_end, onbc, :Glu))

# Basic voltage overlays: all PCs and all ON bipolars
pc_names = sort([name for (name, c) in model.cells if c.cell_type == :PC], by=string)
onbc_names = sort([name for (name, c) in model.cells if c.cell_type == :ONBC], by=string)

fig = Figure(size=(900, 700))
ax_pc = Axis(fig[1, 1], title="Photoreceptor Voltages", xlabel="Time (ms)", ylabel="V (mV)")
ax_on = Axis(fig[2, 1], title="ON Bipolar Voltages", xlabel="Time (ms)", ylabel="V (mV)")

for name in pc_names
    cell = model.cells[name]
    vtrace = [get_out(ui, cell, :V) for ui in sol.u]
    lines!(ax_pc, sol.t, vtrace, linewidth=2, label=String(name))
end

for name in onbc_names
    cell = model.cells[name]
    vtrace = [get_out(ui, cell, :V) for ui in sol.u]
    lines!(ax_on, sol.t, vtrace, linewidth=2, label=String(name))
end

axislegend(ax_pc, position=:rb)
axislegend(ax_on, position=:rb)

mkpath("examples/plots")
save("examples/plots/run_retinal_column_voltages.png", fig)
