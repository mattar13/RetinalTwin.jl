using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics
using ElectroPhysiology
import ElectroPhysiology: parse_stimulus_name, calculate_photons, percent_to_photons_eq, nd_filter

println("=" ^ 70)
println("A-wave simulation workflow (photoreceptor-dominant BaCl + LAP4)")
println("=" ^ 70)

#%% 1) Load real BaCl+LAP4 ERG data via ElectroPhysiology.jl ────────────────
response_window = (0.5, 1.5)
t_post = 6.0   
erg_dir = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods"
erg_files = parseABF(erg_dir)
real_dataset = openERGData(erg_files, t_post = 9.74)
real_t = real_dataset.t
dt = real_t[2] - real_t[1]

stimulus_table = ElectroPhysiology.stimulus_to_table(real_dataset)
save_stimulus_csv(joinpath(@__DIR__, "inputs", "stimulus_table.csv"), real_dataset)

# Load the stimulus model from the saved CSV
stimulus_model = load_stimulus_table(joinpath(@__DIR__, "inputs", "stimulus_table.csv"))
intensity_levels = [s.intensity for s in stimulus_model]
n_sweeps = length(stimulus_model)

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], xlabel="Time (ms)", ylabel="ERG")
for i in 1:n_sweeps
    lines!(ax, real_t[i], real_traces[i], label="")
end
display(fig)

# -----------------------------------------------------------------------------
#%% 3) Build model + parameter set for BaCl+LAP4-like blocked components
#
# Goal: photoreceptor-dominant response (a-wave focused), while preserving the
# full ERG depth map and cell architecture.
# - ON bipolar signaling blocked (LAP4-like) via TRPM1 conductance.
# - OFF bipolar signaling blocked via iGluR conductance.
# - Müller glia component blocked (BaCl-like) via Kir conductances.
# -----------------------------------------------------------------------------
load_param_fn = joinpath(@__DIR__, "inputs", "retinal_params.csv")
params_dict = load_all_params(csv_path = load_param_fn, editable = true)

params_dict[:ONBC][:g_TRPM1] = 0.0
params_dict[:OFFBC][:g_iGluR] = 0.0
params_dict[:MULLER][:g_Kir_end] = 0.0
params_dict[:MULLER][:g_Kir_stalk] = 0.0

params = dict_to_namedtuple(params_dict)

col_path = joinpath(@__DIR__, "inputs", "photoreceptor_column.json")
model, u0 = load_mapping(col_path)

println("Cells in model: ", ordered_cells(model))
println("Total states: ", length(u0))

u0 = dark_adapt(model, u0, params; abstol=1e-6, reltol=1e-4)

#%% 4) IR protocol and ERG extraction for each flash intensity

stim_start = 0.0

# save outputs
solutions = Vector{Any}(undef, length(intensity_levels))
erg_traces = Vector{Vector{Float64}}(undef, length(intensity_levels))
peak_amp = fill(NaN, length(intensity_levels))

tspan = (0.0, real_dataset.t[end])
dt = 0.01
t_rng = tspan[1]:dt:tspan[2]

t, erg_traces, solutions, peak_amp = simulate_erg(model, u0, params, stimulus_model; tspan=tspan, dt=dt);

#%% Calculate the residual
# Full-waveform MSE and residuals aligned to the exact MSE indexing strategy.
trace_mse = RetinalTwin.mean_squared_error(t_rng, erg_traces, real_t, real_traces)
println("Trace MSE (full waveform): ", trace_mse)

trace_residual_abs = RetinalTwin.residual_traces(
    t_rng, erg_traces, real_t, real_traces; absolute=true
)
isempty(trace_residual_abs) && error("Could not compute residual traces: invalid trace/time inputs.")

ir_residual = real_amp .- peak_amp


#%% 6) Plot simulated column (left) and real-data column (right)
 fig = Figure(size=(1200, 800))
log_intensity_levels = log10.(intensity_levels)
logI_min, logI_max = extrema(log_intensity_levels)
trace_cmap = cgrad(:viridis)
trace_colors = get.(Ref(trace_cmap), (log_intensity_levels .- logI_min) ./ (logI_max - logI_min + eps()))
resid_cmap = cgrad(:Reds)
resid_colors = get.(Ref(resid_cmap), 0.35 .+ 0.60 .* ((log_intensity_levels .- logI_min) ./ (logI_max - logI_min + eps())))

ax_sim_trace = Axis(
    fig[1, 1],
    xlabel="Time (ms)",
    ylabel="ERG (a.u.)",
    title="Simulated BaCl + LAP4 ERG",
)
for (i, I) in enumerate(intensity_levels)
    lines!(ax_sim_trace, t_rng, erg_traces[i], color=trace_colors[i], linewidth=2.0, label="I=$(I)")
    vspan!(ax_sim_trace, stim_start, stim_start + stimulus_model[i].duration / 10.0, color=(:gold, 0.15))
end
# axislegend(ax_sim_trace, position=:rb)

# Simulated IR + Hill fit (bottom-left) -----------------------------------------
ax_sim_ir = Axis(
    fig[1, 2],
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
    fig[2, 1],
    xlabel="Time (ms)",
    ylabel="ERG (a.u.)",
    title="Real BaCl + LAP4 ERG",
)
for i in 1:n_sweeps
    lines!(ax_real_trace, real_t[i], real_traces[i], color=trace_colors[i], linewidth=2.0, label="")
    vspan!(ax_real_trace, stim_start, stim_start + stimulus_model[i].duration / 10.0, color=(:gold, 0.15))
end
# axislegend(ax_real_trace, position=:rb)

Colorbar(
    fig[1, 3],
    colormap=:viridis,
    limits=(logI_min, logI_max),
    label="log10 photons",
)

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

# Residual traces (third-row left) --------------------------------------------
ax_trace_resid = Axis(
    fig[3, 1],
    xlabel="Time (ms)",
    ylabel="|Residual|",
    title="Trace residuals in fit window (|real - sim|)",
)

for i in 1:n_sweeps
    band!(
        ax_trace_resid,
        real_t[i],
        zeros(length(real_t[i])),
        trace_residual_abs[i],
        color=(resid_colors[i], 0.50),
    )
end 
# hline!(ax_trace_resid, [0.0], color=(:black, 0.4), linewidth=1.0)

# IR residuals (third-row right) -----------------------------------------------
ax_ir_resid = Axis(
    fig[3, 2],
    xlabel="Flash intensity (photon flux)",
    ylabel="Residual (real - sim)",
    xscale=log10,
    title="IR residuals",
)
lines!(ax_ir_resid, intensity_levels, ir_residual, color=:red, linewidth=2.5)
scatter!(ax_ir_resid, intensity_levels, ir_residual, color=:red, markersize=8)
# hline!(ax_ir_resid, [0.0], color=(:black, 0.4), linewidth=1.0, linestyle=:dash)

linkxaxes!(ax_sim_ir, ax_real_ir, ax_ir_resid)
linkxaxes!(ax_sim_trace, ax_real_trace, ax_trace_resid)


#save the figure
save(joinpath(@__DIR__, "plots", "awave_ir_sim_vs_real.png"), fig)
display(fig)