using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics
using ElectroPhysiology

println("=" ^ 70)
println("A-wave simulation workflow (photoreceptor-dominant BaCl + LAP4)")
println("=" ^ 70)

##%% Set up some overall parameters
response_window = (0.5, 1.5)

# -----------------------------------------------------------------------------
#%% 1) Open real data and extract trial traces
# -----------------------------------------------------------------------------
stimulus_intensity_levels = [
    (0, 1.0, 4000.0) #nd0 1ms flash
    (0, 2.0, 4000.0) #nd0 2ms flash
    (1, 1.0, 400.0) #nd1 1ms flash
    (1, 2.0, 400.0) #nd1 2ms flash
    (1, 4.0, 400.0) #nd1 4ms flash
    (2, 1.0, 40.0) #nd2 1ms flash
    (2, 2.0, 40.0) #nd2 2ms flash
    (2, 4.0, 40.0) #nd2 4ms flash
    (3, 1.0, 4.0) #nd3 1ms flash
    (3, 2.0, 4.0) #nd3 2ms flash
    (3, 4.0, 4.0) #nd3 4ms flash 
    (4, 1.0, 0.4) #nd4 1ms flash photon_flux=0.4
    (4, 2.0, 0.4) #nd4 2ms flash 
    (4, 4.0, 0.4) #nd4 4ms flash 
]

intensity_levels = [flash_duration * photon_flux for (_, flash_duration, photon_flux) in stimulus_intensity_levels]
sorted_int_idx = sortperm(intensity_levels)

intensity_levels = intensity_levels[sorted_int_idx]
stimulus_intensity_levels = stimulus_intensity_levels[sorted_int_idx]

erg_dir = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods"
erg_files = parseABF(erg_dir)[sorted_int_idx]

stim_channel = "IN 7"
real_dataset = readABF(
    erg_files;
    stimulus_name=stim_channel,
    align_by_stimulus=true,
    t_pre=0.1,
    t_post=6.0,
)

n_sweeps = eachtrial(real_dataset) |> length
real_traces = Vector{Vector{Float64}}(undef, n_sweeps)
real_t = Vector{Vector{Float64}}(undef, n_sweeps)
real_amp = fill(NaN, n_sweeps)

for (i, sweep) in enumerate(eachtrial(real_dataset))
    t = sweep.t
    y = sweep.data_array[1,:,1] * 1000.0
    # println(sweep.HeaderDict)
    y .-= y[1] #baseline subtract
    real_traces[i] = y
    real_t[i] = t
    first_peak_idx = findfirst(t .>= response_window[1])
    last_peak_idx = findfirst(t .>= response_window[2])
    real_amp[i] = -minimum(y[first_peak_idx:last_peak_idx])
end

# -----------------------------------------------------------------------------
#%% 2) Build model + parameter set for BaCl+LAP4-like blocked components
#
# Goal: photoreceptor-dominant response (a-wave focused), while preserving the
# full ERG depth map and cell architecture.
# - ON bipolar signaling blocked (LAP4-like) via TRPM1 conductance.
# - OFF bipolar signaling blocked via iGluR conductance.
# - Müller glia component blocked (BaCl-like) via Kir conductances.
# -----------------------------------------------------------------------------
params_dict = default_retinal_params(editable=true)

# ON bipolar block (b-wave component)
params_dict[:ON_BIPOLAR_PARAMS][:g_TRPM1] = 0.0

# OFF bipolar block
params_dict[:OFF_BIPOLAR_PARAMS][:g_iGluR] = 0.0

# Müller glia block (BaCl-like)
params_dict[:MULLER_PARAMS][:g_Kir_end] = 0.0
params_dict[:MULLER_PARAMS][:g_Kir_stalk] = 0.0

params = dict_to_namedtuple(params_dict)

pc_coords = square_grid_coords(16)
model, u0 = build_column(
    nPC=16,
    nHC=4,
    nONBC=4,
    onbc_pool_size=4,
    nOFFBC=4,
    nA2=4,
    nGC=1,
    nMG=4,
    pc_coords=pc_coords,
)

println("Cells in model: ", ordered_cells(model))
println("Total states: ", length(u0))

# -----------------------------------------------------------------------------
# 3) Dark adapt initial condition
# -----------------------------------------------------------------------------
tspan_dark = (0.0, 2000.0)
stim_dark = RetinalTwin.make_uniform_flash_stimulus(photon_flux=0.0)
prob_dark = ODEProblem(model, u0, tspan_dark, (params, stim_dark))
sol_dark = solve(
    prob_dark,
    Rodas5();
    save_everystep=false,
    save_start=false,
    save_end=true,
    abstol=1e-6,
    reltol=1e-4,
)
u0_dark = sol_dark.u[end]

# -----------------------------------------------------------------------------
# 4) IR protocol and ERG extraction for each flash intensity
# -----------------------------------------------------------------------------
stim_start = 0.0

# save outputs
solutions = Vector{Any}(undef, length(intensity_levels))
erg_traces = Vector{Vector{Float64}}(undef, length(intensity_levels))
peak_amp = fill(NaN, length(intensity_levels))

tspan = (0.0, 6.0)
dt = 0.01
t_rng = tspan[1]:dt:tspan[2]

for (i, (nd, flash_duration, photon_flux)) in enumerate(stimulus_intensity_levels)
    stim = RetinalTwin.make_uniform_flash_stimulus(
        stim_start=stim_start,
        stim_end=stim_start + flash_duration/1000.0,
        photon_flux=photon_flux,
    )

    prob = ODEProblem(model, u0_dark, tspan, (params, stim))
    sol = solve(
        prob,
        Rodas5();
        tstops=[stim_start, stim_start + flash_duration/1000.0],
        abstol=1e-6,
        reltol=1e-4,
    )

    t_erg, erg = compute_field_potential(model, params, sol; dt = dt)
    solutions[i] = sol
    erg_traces[i] = erg

    response_idx = findall(t -> response_window[1] <= t <= response_window[2], t_rng)
    peak_amp[i] = -minimum(erg[response_idx])

    println(
        "Intensity $(photon_flux) $(flash_duration)ms: retcode=$(sol.retcode), " *
        "a-wave amp=$(round(peak_amp[i], digits=4))"
    )
end

#We want to sort the peak_awave
# -----------------------------------------------------------------------------
#%% 6) Plot simulated column (left) and real-data column (right)
# -----------------------------------------------------------------------------
fig = Figure(size=(1600, 820))
palette = cgrad(:viridis, length(intensity_levels), categorical=true)

# Simulated traces (top-left) --------------------------------------------------
ax_sim_trace = Axis(
    fig[1, 1],
    xlabel="Time (ms)",
    ylabel="ERG (a.u.)",
    title="Simulated BaCl + LAP4 ERG",
)
for (i, I) in enumerate(intensity_levels)
    lines!(ax_sim_trace, t_rng, erg_traces[i], color=palette[i], linewidth=2.0, label="I=$(I)")
    vspan!(ax_sim_trace, stim_start, stim_start + stimulus_intensity_levels[i][2]/1000.0, color=(:gold, 0.15))
end
axislegend(ax_sim_trace, position=:rb)

# Simulated IR + Hill fit (bottom-left) -----------------------------------------
ax_sim_ir = Axis(
    fig[2, 1],
    xlabel="Flash intensity (photon flux)",
    ylabel="A-wave amplitude (-minimum ERG)",
    xscale=log10,
    title="Simulated IR + Hill fit",
)
lines!(ax_sim_ir, intensity_levels, peak_amp, color=:black, linewidth=2, label="Simulated a-wave amplitude")
scatter!(ax_sim_ir, intensity_levels, peak_amp, color=:crimson, markersize=10)

sim_fit = fit_hill_ir(intensity_levels, peak_amp)
if sim_fit.ok
    I_fit = exp10.(range(log10(minimum(intensity_levels)), log10(maximum(intensity_levels)); length=300))
    y_fit = [hill_ir(Ii, sim_fit.A, sim_fit.K, sim_fit.n) for Ii in I_fit]
    lbl = "Hill fit: K=$(round(sim_fit.K, sigdigits=3)), n=$(round(sim_fit.n, sigdigits=3))"
    lines!(ax_sim_ir, I_fit, y_fit, color=:dodgerblue, linewidth=2.5, linestyle=:dash, label=lbl)
    println("Simulated Hill fit: A=$(sim_fit.A), K_Ihalf=$(sim_fit.K), n=$(sim_fit.n), sse=$(sim_fit.sse)")
else
    println("Simulated Hill fit failed.")
end
axislegend(ax_sim_ir, position=:rb)

# Real traces (top-right) -------------------------------------------------------
ax_real_trace = Axis(
    fig[1, 2],
    xlabel="Time (ms)",
    ylabel="ERG (a.u.)",
    title="Real BaCl + LAP4 ERG",
)
for i in 1:n_sweeps
    lines!(ax_real_trace, real_t[i], real_traces[i], color=palette[i], linewidth=2.0, label="") 
    vspan!(ax_real_trace, stim_start, stim_start + stimulus_intensity_levels[i][2]/1000.0, color=(:gold, 0.15))
end
axislegend(ax_real_trace, position=:rb)

# Real IR + Hill fit (bottom-right)
ax_real_ir = Axis(
    fig[2, 2],
    xlabel="Flash intensity (photon flux)",
    ylabel="A-wave amplitude (-minimum ERG)",
    xscale=log10,
    title="Real IR + Hill fit",
)

lines!(ax_real_ir, intensity_levels, sort(real_amp), color=:black, linewidth=2, label="Real a-wave amplitude")
scatter!(ax_real_ir, intensity_levels, sort(real_amp), color=:crimson, markersize=10)

real_fit = fit_hill_ir(intensity_levels, sort(real_amp))
if real_fit.ok
    I_fit = exp10.(range(log10(minimum(intensity_levels)), log10(maximum(intensity_levels)); length=300))
    y_fit = [hill_ir(Ii, real_fit.A, real_fit.K, real_fit.n) for Ii in I_fit]
    lbl = "Hill fit: K=$(round(real_fit.K, sigdigits=3)), n=$(round(real_fit.n, sigdigits=3))"
    lines!(ax_real_ir, I_fit, y_fit, color=:dodgerblue, linewidth=2.5, linestyle=:dash, label=lbl)
    println("Real-data Hill fit: A=$(real_fit.A), K_Ihalf=$(real_fit.K), n=$(real_fit.n), sse=$(real_fit.sse)")
end
axislegend(ax_real_ir, position=:rb)


#save the figure
save(joinpath(@__DIR__, "plots", "awave_ir_sim_vs_real.png"), fig)
display(fig)

# -----------------------------------------------------------------------------
#%% 7) Optional IR summary table (sim + real)
# -----------------------------------------------------------------------------
out_csv = joinpath(@__DIR__, "data", "awave_ir_simulation.csv")
open(out_csv, "w") do io
    println(io, "dataset,intensity,awave_peak_min_erg,awave_amplitude")
    for (I, peak_min, amp) in zip(intensity_levels, peak_amp)
        println(io, "simulated,$(I),$(peak_min),$(amp)")
    end
    for (I, peak_min, amp) in zip(real_intensity_levels, real_awave_amp)
        println(io, "real,$(I),$(peak_min),$(amp)")
    end
end
println("Saved IR summary CSV: ", out_csv)
