using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics

println("=" ^ 70)
println("A-wave simulation workflow (photoreceptor-dominant BaCl + LAP4)")
println("=" ^ 70)

# -----------------------------------------------------------------------------
#%% 1) Input file paths (all from examples/inputs/)
# -----------------------------------------------------------------------------
input_dir    = joinpath(@__DIR__, "inputs", "default")
structure_fn = joinpath(input_dir, "photoreceptor_column.json")
params_fn    = joinpath(input_dir, "retinal_params.csv")
stimulus_fn  = joinpath(input_dir, "stimulus_table.csv")
depth_fn     = joinpath(input_dir, "erg_depth_map.csv")

# -----------------------------------------------------------------------------
#%% 2) Load inputs + apply BaCl+LAP4 blockers
# -----------------------------------------------------------------------------
model, u0 = load_mapping(structure_fn)
stimulus_table = load_stimulus_table(stimulus_fn)

params_dict = load_all_params(csv_path = params_fn, editable = true)
params_dict[:ONBC][:g_TRPM1]      = 0.0
params_dict[:OFFBC][:g_iGluR]     = 0.0
params_dict[:MULLER][:g_Kir_end]   = 0.0
params_dict[:MULLER][:g_Kir_stalk] = 0.0
params = dict_to_namedtuple(params_dict)

println("Cells in model: ", ordered_cells(model))
println("Total states:   ", length(u0))
println("Stimuli:        ", length(stimulus_table), " sweeps")

# -----------------------------------------------------------------------------
#%% 3) Simulate ERG (auto dark-adapts)
# -----------------------------------------------------------------------------
tspan = (0.0, 6.0)
dt = 0.01

t, erg_traces, solutions, peak_amps = simulate_erg(;
    structure = (model, u0),
    params = params,
    stimulus_table = stimulus_table,
    depth_csv = depth_fn,
    tspan = tspan, dt = dt,
    verbose = true,
)

# -----------------------------------------------------------------------------
#%% 4) Load real ERG data for comparison
# -----------------------------------------------------------------------------
using ElectroPhysiology
import ElectroPhysiology: save_stimulus_csv

response_window = (0.5, 1.5)
erg_dir = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods"
erg_files = parseABF(erg_dir)
real_dataset = openERGData(erg_files, t_post = 9.74)

# Re-save stimulus from real data (keeps inputs/ in sync)
save_stimulus_csv(stimulus_fn, real_dataset)

n_sweeps = length(stimulus_table)
real_traces = Vector{Vector{Float64}}(undef, n_sweeps)
real_t_vec  = Vector{Vector{Float64}}(undef, n_sweeps)
real_amp    = fill(NaN, n_sweeps)

for (i, sweep) in enumerate(eachtrial(real_dataset))
    t_s = sweep.t
    y = sweep.data_array[1,:,1] * 1000.0
    real_traces[i] = y
    real_t_vec[i]  = t_s
    first_peak_idx = findfirst(t_s .>= response_window[1])
    last_peak_idx  = findfirst(t_s .>= response_window[2])
    real_amp[i] = -minimum(y[first_peak_idx:last_peak_idx])
end

# -----------------------------------------------------------------------------
#%% 5) Compute residuals
# -----------------------------------------------------------------------------
t_rng = tspan[1]:dt:tspan[2]
trace_mse = RetinalTwin.mean_squared_error(t_rng, erg_traces, real_t_vec, real_traces)
println("Trace MSE (full waveform): ", trace_mse)

trace_residual_abs = RetinalTwin.residual_traces(
    t_rng, erg_traces, real_t_vec, real_traces; absolute=true
)
isempty(trace_residual_abs) && error("Could not compute residual traces.")

ir_residual = real_amp .- peak_amps
intensity_levels = [s.intensity for s in stimulus_table]

# -----------------------------------------------------------------------------
#%% 6) Plot: simulated vs real ERG
# -----------------------------------------------------------------------------
log_I = log10.(intensity_levels)
logI_min, logI_max = extrema(log_I)
trace_cmap = cgrad(:viridis)
sweep_colors = [get(trace_cmap, (li - logI_min) / (logI_max - logI_min + eps())) for li in log_I]
resid_cmap = cgrad(:Reds)
resid_colors = [get(resid_cmap, 0.35 + 0.60 * ((li - logI_min) / (logI_max - logI_min + eps()))) for li in log_I]

fig = Figure(size=(1200, 800))

# Simulated traces (top-left)
ax_sim_trace = Axis(fig[1, 1], xlabel="Time (s)", ylabel="ERG (a.u.)",
    title="Simulated BaCl + LAP4 ERG")
for (i, s) in enumerate(stimulus_table)
    lines!(ax_sim_trace, collect(t_rng), erg_traces[i], color=sweep_colors[i], linewidth=2.0)
end

# Simulated IR + Hill fit (top-right)
ax_sim_ir = Axis(fig[1, 2], xlabel="Flash intensity (photon flux)",
    ylabel="A-wave amplitude", xscale=log10, title="Simulated IR + Hill fit")
lines!(ax_sim_ir, intensity_levels, peak_amps, color=:black, linewidth=2)
scatter!(ax_sim_ir, intensity_levels, peak_amps, color=:crimson, markersize=10)

sim_fit = fit_hill_ir(intensity_levels, peak_amps)
if sim_fit.ok
    I_fit = exp10.(range(log10(minimum(intensity_levels)), log10(maximum(intensity_levels)); length=300))
    y_fit = [hill_ir(Ii, sim_fit.A, sim_fit.K, sim_fit.n) for Ii in I_fit]
    lbl = "Hill: K=$(round(sim_fit.K, sigdigits=3)), n=$(round(sim_fit.n, sigdigits=3))"
    lines!(ax_sim_ir, I_fit, y_fit, color=:dodgerblue, linewidth=2.5, linestyle=:dash, label=lbl)
    println("Simulated Hill fit: A=$(sim_fit.A), K=$(sim_fit.K), n=$(sim_fit.n)")
end
axislegend(ax_sim_ir, position=:rb)

# Real traces (middle-left)
ax_real_trace = Axis(fig[2, 1], xlabel="Time (s)", ylabel="ERG (a.u.)",
    title="Real BaCl + LAP4 ERG")
for i in 1:n_sweeps
    lines!(ax_real_trace, real_t_vec[i], real_traces[i], color=sweep_colors[i], linewidth=2.0)
end

Colorbar(fig[1:2, 3], colormap=:viridis, limits=(logI_min, logI_max), label="log10 intensity")

# Real IR + Hill fit (middle-right)
ax_real_ir = Axis(fig[2, 2], xlabel="Flash intensity (photon flux)",
    ylabel="A-wave amplitude", xscale=log10, title="Real IR + Hill fit")
sorted_amp = sort(real_amp)
lines!(ax_real_ir, intensity_levels, sorted_amp, color=:black, linewidth=2)
scatter!(ax_real_ir, intensity_levels, sorted_amp, color=:crimson, markersize=10)

real_fit = fit_hill_ir(intensity_levels, sorted_amp)
if real_fit.ok
    I_fit = exp10.(range(log10(minimum(intensity_levels)), log10(maximum(intensity_levels)); length=300))
    y_fit = [hill_ir(Ii, real_fit.A, real_fit.K, real_fit.n) for Ii in I_fit]
    lbl = "Hill: K=$(round(real_fit.K, sigdigits=3)), n=$(round(real_fit.n, sigdigits=3))"
    lines!(ax_real_ir, I_fit, y_fit, color=:dodgerblue, linewidth=2.5, linestyle=:dash, label=lbl)
    println("Real Hill fit: A=$(real_fit.A), K=$(real_fit.K), n=$(real_fit.n)")
end
axislegend(ax_real_ir, position=:rb)

# Residual traces (bottom-left)
ax_trace_resid = Axis(fig[3, 1], xlabel="Time (s)", ylabel="|Residual|",
    title="Trace residuals (|real - sim|)")
for i in 1:n_sweeps
    band!(ax_trace_resid, real_t_vec[i], zeros(length(real_t_vec[i])),
        trace_residual_abs[i], color=(resid_colors[i], 0.50))
end

# IR residuals (bottom-right)
ax_ir_resid = Axis(fig[3, 2], xlabel="Flash intensity (photon flux)",
    ylabel="Residual (real - sim)", xscale=log10, title="IR residuals")
lines!(ax_ir_resid, intensity_levels, ir_residual, color=:red, linewidth=2.5)
scatter!(ax_ir_resid, intensity_levels, ir_residual, color=:red, markersize=8)

linkxaxes!(ax_sim_ir, ax_real_ir, ax_ir_resid)
linkxaxes!(ax_sim_trace, ax_real_trace, ax_trace_resid)

save(joinpath(@__DIR__, "plots", "awave_ir_sim_vs_real.png"), fig)
display(fig)
