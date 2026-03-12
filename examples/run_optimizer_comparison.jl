using Revise
using RetinalTwin
using DifferentialEquations
using CairoMakie
using Statistics
using Dates

println("=" ^ 70)
println("Optimizer comparison — a-wave fitting across methods")
println("=" ^ 70)

# -----------------------------------------------------------------------------
#%% 1) Input file paths
# -----------------------------------------------------------------------------
input_dir    = joinpath(@__DIR__, "inputs", "default")
structure_fn = joinpath(input_dir, "photoreceptor_column.json")
params_fn    = joinpath(input_dir, "retinal_params.csv")
stimulus_fn  = joinpath(input_dir, "stimulus_table.csv")
depth_fn     = joinpath(input_dir, "erg_depth_map.csv")

# -----------------------------------------------------------------------------
#%% 2) Load inputs + apply blockers
# -----------------------------------------------------------------------------
model, u0 = load_mapping(structure_fn)
stimulus_table = load_stimulus_table(stimulus_fn)

params_dict = load_all_params(csv_path = params_fn, editable = true)
params_dict[:ONBC][:g_TRPM1]      = 0.0
params_dict[:OFFBC][:g_iGluR]     = 0.0
params_dict[:MULLER][:g_Kir_end]   = 0.0
params_dict[:MULLER][:g_Kir_stalk] = 0.0
params = dict_to_namedtuple(params_dict)

println("Cells: ", ordered_cells(model))
println("States: ", length(u0))
println("Stimuli: ", length(stimulus_table), " sweeps")

# -----------------------------------------------------------------------------
#%% 3) Load real ERG data
# -----------------------------------------------------------------------------
using ElectroPhysiology
import ElectroPhysiology: save_stimulus_csv

response_window = (0.5, 1.5)
erg_dir = raw"F:\ERG\Retinoschisis\2019_03_12_AdultWT\Mouse1_Adult_WT\BaCl_LAP4\Rods"
erg_files = parseABF(erg_dir)
real_dataset = openERGData(erg_files, t_post = 9.74)

save_stimulus_csv(stimulus_fn, real_dataset)
stimulus_table = load_stimulus_table(stimulus_fn)

n_sweeps = eachtrial(real_dataset) |> length
real_traces = Vector{Vector{Float64}}(undef, n_sweeps)
real_t_vec  = Vector{Vector{Float64}}(undef, n_sweeps)

for (i, sweep) in enumerate(eachtrial(real_dataset))
    real_t_vec[i]  = sweep.t
    real_traces[i] = sweep.data_array[1,:,1] * 1000.0
end

# -----------------------------------------------------------------------------
#%% 4) Run each optimizer with the same budget
# -----------------------------------------------------------------------------
tspan = (0.0, real_dataset.t[end])
dt = 0.01
checkpoint_base = joinpath(@__DIR__, "checkpoints", "optimizer_comparison")
phase1_iters = 300   # same budget for all

optimizers = [:cma_es, :nelder_mead, :particle_swarm, :samin]
optimizer_labels = Dict(
    :cma_es         => "CMA-ES",
    :nelder_mead    => "Nelder-Mead",
    :particle_swarm => "Particle Swarm",
    :samin          => "Simulated Annealing",
)
optimizer_colors = Dict(
    :cma_es         => :royalblue,
    :nelder_mead    => :crimson,
    :particle_swarm => :forestgreen,
    :samin          => :darkorange,
)

results = Dict{Symbol, Any}()

for opt in optimizers
    println("\n" * "=" ^ 70)
    println("Running $(optimizer_labels[opt])...")
    println("=" ^ 70)

    try
        r = fit_erg(
            model, u0, params;
            cell_types = [:PHOTO],
            stimuli = stimulus_table,
            real_t = real_t_vec,
            real_traces = real_traces,
            time_window = response_window,
            tspan = tspan,
            dt = dt,
            optimizer = opt,
            phase1_iterations = phase1_iters,
            run_lbfgs = false,   # compare phase 1 only
            checkpoint_every = 0,
            checkpoint_dir = checkpoint_base,
            verbose = true,
        )
        results[opt] = r
        println("  Final loss: $(round(r.loss, sigdigits=6))")
    catch e
        println("  $(optimizer_labels[opt]) failed: $e")
    end
end

# -----------------------------------------------------------------------------
#%% 5) Convergence comparison figure
# -----------------------------------------------------------------------------
fig = Figure(size=(1400, 900))

# Panel 1: Best loss vs evaluations
ax1 = Axis(fig[1, 1],
    xlabel = "Objective evaluations",
    ylabel = "Best loss",
    yscale = log10,
    title  = "Convergence: best loss vs evaluations",
)
for opt in optimizers
    haskey(results, opt) || continue
    h = results[opt].loss_history
    isempty(h) && continue
    evals = [x.eval for x in h]
    best  = [x.best_loss for x in h]
    lines!(ax1, evals, best, color=optimizer_colors[opt], linewidth=2.5,
        label=optimizer_labels[opt])
end
axislegend(ax1, position=:rt)

# Panel 2: Best loss vs wall-clock time
ax2 = Axis(fig[1, 2],
    xlabel = "Wall-clock time (s)",
    ylabel = "Best loss",
    yscale = log10,
    title  = "Convergence: best loss vs time",
)
for opt in optimizers
    haskey(results, opt) || continue
    h = results[opt].loss_history
    isempty(h) && continue
    elapsed = [x.elapsed for x in h]
    best    = [x.best_loss for x in h]
    lines!(ax2, elapsed, best, color=optimizer_colors[opt], linewidth=2.5,
        label=optimizer_labels[opt])
end
axislegend(ax2, position=:rt)

# Panel 3: Final loss bar chart
ax3 = Axis(fig[2, 1],
    xlabel = "Optimizer",
    ylabel = "Final loss",
    yscale = log10,
    title  = "Final loss comparison",
    xticklabelrotation = π/6,
)
bar_opts = Symbol[opt for opt in optimizers if haskey(results, opt)]
bar_x = 1:length(bar_opts)
bar_y = [results[opt].loss for opt in bar_opts]
bar_c = [optimizer_colors[opt] for opt in bar_opts]
barplot!(ax3, collect(bar_x), bar_y, color=bar_c)
ax3.xticks = (collect(bar_x), [optimizer_labels[opt] for opt in bar_opts])

# Panel 4: Summary table as text
ax4 = Axis(fig[2, 2], title="Summary")
hidespines!(ax4)
hidedecorations!(ax4)

summary_lines = String[]
push!(summary_lines, "Optimizer           | Final Loss   | Evals | Time (s)")
push!(summary_lines, "-" ^ 60)
for opt in optimizers
    haskey(results, opt) || continue
    r = results[opt]
    h = r.loss_history
    n_evals = isempty(h) ? 0 : h[end].eval
    elapsed = isempty(h) ? 0.0 : h[end].elapsed
    push!(summary_lines,
        "$(rpad(optimizer_labels[opt], 20))| $(lpad(string(round(r.loss, sigdigits=6)), 12)) | $(lpad(n_evals, 5)) | $(lpad(round(elapsed, digits=1), 7))")
end
text!(ax4, 0.02, 0.95, text=join(summary_lines, "\n"),
    fontsize=13, font=:mono, align=(:left, :top))

outdir = joinpath(@__DIR__, "plots")
mkpath(outdir)
save(joinpath(outdir, "optimizer_comparison.png"), fig)
display(fig)

# -----------------------------------------------------------------------------
#%% 6) Save best result as checkpoint
# -----------------------------------------------------------------------------
if !isempty(results)
    best_opt = argmin(Dict(opt => results[opt].loss for opt in keys(results)))
    best_result = results[best_opt]
    println("\nBest optimizer: $(optimizer_labels[best_opt]) (loss=$(round(best_result.loss, sigdigits=6)))")

    final_dir = joinpath(checkpoint_base, "best_$(best_opt)_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS"))")
    mkpath(final_dir)
    save_fitted_params_csv(best_result, joinpath(final_dir, "retinal_params.csv");
        template_csv = params_fn)
    save_fit_datasheet(best_result, joinpath(final_dir, "awave_fit_params.csv"))
    for (src, dst_name) in [
        (structure_fn, "structure.json"),
        (stimulus_fn,  "stimulus_table.csv"),
        (depth_fn,     "erg_depth_map.csv"),
    ]
        isfile(src) && cp(src, joinpath(final_dir, dst_name); force=true)
    end
    println("Best result saved to: $final_dir")
end
