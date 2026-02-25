using Revise
using RetinalTwin
using DifferentialEquations

println("=" ^ 60)
println("RetinalColumnModel Example (Full Build)")
println("=" ^ 60)

#%% --------- Build/load model ---------
params = load_all_params(editable=true)
map_path = joinpath(@__DIR__, "data", "column_map.json")
if isfile(map_path)
    model, u0 = load_mapping(map_path)
else
    pc_coords = square_grid_coords(16)
    model, u0 = build_column(; nPC=16, nHC=4, nONBC=4, nOFFBC=0, nA2=4, nGC=1, nMG=4, pc_coords=pc_coords)
    save_mapping(map_path, model, u0)
end

println("Cells: ", ordered_cells(model))
println("Total states: ", length(u0))

# Dark adaptation
tspan_dark = (0.0, 2000.0)
stim_dark = RetinalTwin.make_uniform_flash_stimulus(photon_flux=0.0)
sim_params = dict_to_namedtuple(params)
prob_dark = ODEProblem(model, u0, tspan_dark, (sim_params, stim_dark))
sol_dark = solve(prob_dark, Rodas5(); save_everystep=false, save_start=false, save_end=true, abstol=1e-6, reltol=1e-4)
u0 = sol_dark.u[end]

# Uniform flash run
stim_start = 80.0
stim_end = 180.0
photon_flux = 0.10

selected_stimulus = RetinalTwin.make_uniform_flash_stimulus(
    stim_start=stim_start,
    stim_end=stim_end,
    photon_flux=photon_flux,
)

fig = plot_cell_layout_3d(
    model;
    z_stim=0.0,
    stimulus_func=selected_stimulus,
    stimulus_time=(stim_start + stim_end) / 2,
    stimulus_grid_n=61,
    savepath=joinpath(@__DIR__, "plots", "uniform_flash_column_layout_3d.png"),
)
display(fig)

#%% --------- Run the model ---------
# params[:ON_BIPOLAR_PARAMS][:g_TRPM1] = 0.0 # Disable TRPM1 channel for no b-wave
# params[:OFF_BIPOLAR_PARAMS][:g_iGluR] = 0.0 # Disable iGluR channel for no d-wave
params[:MULLER_PARAMS][:g_Kir_end] = 0.0 # Disable Kir channel for no a-wave
params[:MULLER_PARAMS][:g_Kir_stalk] = 0.0 # Disable Kir channel for no a-wave

sim_params = dict_to_namedtuple(params)

tspan = (0.0, 1200.0)
prob = ODEProblem(model, u0, tspan, (sim_params, selected_stimulus))
@time sol = solve(prob, Rodas5(); tstops = [stim_start, stim_end], saveat=1.0, abstol=1e-6, reltol=1e-4)

println("Solver status: ", sol.retcode)
println("Saved time points: ", length(sol.t))


fig = plot_all_cells_v_ca_release(
    sol,
    model;
    stim_start=stim_start,
    stim_end=stim_end,
    savepath=joinpath(@__DIR__, "plots", "uniform_flash_all_cells_only_a_wave.png"),
)

#%% --------- Compute field potential ---------
t, erg = compute_field_potential(model, sim_params, sol)

fig_erg = Figure(size=(1100, 650))
ax_erg = Axis(fig_erg[1, 1], xlabel="Time (ms)", ylabel="ERG (mV)")
lines!(ax_erg, t, erg)
display(fig_erg)
