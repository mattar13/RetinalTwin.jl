# ============================================================
# plots.jl — Plotting utilities using CairoMakie
# ============================================================

using CairoMakie

"""
    plot_erg(result; show_components=true, time_range=nothing)

Plot the total ERG and optionally decomposed components.
`result` is the NamedTuple from `simulate_flash`.
"""
function plot_erg(result; show_components::Bool=true, time_range=nothing)
    t = result.t
    erg = result.erg
    comps = result.erg_components

    # Apply time range filter
    if time_range !== nothing
        mask = (t .>= time_range[1]) .& (t .<= time_range[2])
        t = t[mask]
        erg = erg[mask]
        comps = Dict(k => v[mask] for (k, v) in comps)
    end

    if show_components
        fig = Figure(size=(900, 800))

        # Total ERG
        ax1 = Axis(fig[1, 1], title="Total ERG", xlabel="Time (ms)", ylabel="Amplitude (a.u.)")
        lines!(ax1, t, erg, color=:black, linewidth=2)
        hlines!(ax1, [0.0], color=:gray, linestyle=:dash)

        # Component decomposition
        ax2 = Axis(fig[2, 1], title="ERG Components", xlabel="Time (ms)", ylabel="Amplitude (a.u.)")

        component_colors = Dict(
            :a_wave => :blue, :b_wave => :red, :d_wave => :orange,
            :OPs => :green, :P3 => :purple, :c_wave => :brown, :ganglion => :gray
        )
        component_labels = Dict(
            :a_wave => "a-wave (PR)", :b_wave => "b-wave (ON-BC)", :d_wave => "d-wave (OFF-BC)",
            :OPs => "OPs (Amacrine)", :P3 => "P3 (Müller)", :c_wave => "c-wave (RPE)",
            :ganglion => "GC"
        )

        for (key, trace) in comps
            col = get(component_colors, key, :black)
            lab = get(component_labels, key, String(key))
            lines!(ax2, t, trace, color=col, label=lab)
        end
        axislegend(ax2, position=:rt)
        hlines!(ax2, [0.0], color=:gray, linestyle=:dash)
    else
        fig = Figure(size=(900, 400))
        ax = Axis(fig[1, 1], title="ERG", xlabel="Time (ms)", ylabel="Amplitude (a.u.)")
        lines!(ax, t, erg, color=:black, linewidth=2)
        hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    end

    return fig
end

"""
    plot_cell_voltages(result; cell_types=[:rod, :on_bc, :a2, :gc])

Plot membrane potential traces for selected cell types.
"""
function plot_cell_voltages(result; cell_types::Vector{Symbol}=[:rod, :on_bc, :a2, :gc],
                            time_range=nothing)
    t = result.t
    voltages = result.cell_voltages

    if time_range !== nothing
        mask = (t .>= time_range[1]) .& (t .<= time_range[2])
        t = t[mask]
    else
        mask = trues(length(t))
    end

    n_panels = length(cell_types)
    fig = Figure(size=(900, 250 * n_panels))

    cell_labels = Dict(
        :rod => "Rod", :cone => "Cone", :hc => "Horizontal Cell",
        :on_bc => "ON-Bipolar", :off_bc => "OFF-Bipolar",
        :a2 => "A2 Amacrine", :gaba_ac => "GABA Amacrine",
        :da_ac => "DA Amacrine", :gc => "Ganglion Cell",
        :muller => "Müller Glia", :rpe => "RPE"
    )

    for (idx, ct) in enumerate(cell_types)
        label = get(cell_labels, ct, String(ct))
        ax = Axis(fig[idx, 1], title=label, xlabel="Time (ms)", ylabel="V (mV)")

        if haskey(voltages, ct)
            V = voltages[ct][mask, :]
            n_cells = size(V, 2)
            for ci in 1:min(n_cells, 5)  # Plot up to 5 cells
                lines!(ax, t, V[:, ci], linewidth=1)
            end
            if n_cells > 5
                # Plot mean for large populations
                lines!(ax, t, vec(mean(V, dims=2)), color=:black, linewidth=2, linestyle=:dash)
            end
        end
    end

    return fig
end

"""
    plot_ops(result; time_window=(180.0, 280.0))

Plot extracted oscillatory potentials.
"""
function plot_ops(result; time_window::Tuple{Float64,Float64}=(180.0, 280.0))
    t = result.t
    erg = result.erg

    ops = extract_ops(erg, collect(t))

    mask = (t .>= time_window[1]) .& (t .<= time_window[2])
    t_w = t[mask]
    ops_w = ops[mask]
    erg_w = erg[mask]

    fig = Figure(size=(900, 500))

    ax1 = Axis(fig[1, 1], title="ERG (OP window)", xlabel="Time (ms)", ylabel="Amplitude")
    lines!(ax1, t_w, erg_w, color=:black, linewidth=1.5)

    ax2 = Axis(fig[2, 1], title="Extracted OPs (75-300 Hz bandpass)",
               xlabel="Time (ms)", ylabel="Amplitude")
    lines!(ax2, t_w, ops_w, color=:green, linewidth=1.5)
    hlines!(ax2, [0.0], color=:gray, linestyle=:dash)

    return fig
end

"""
    plot_intensity_response(results; measure=:b_wave)

Plot intensity-response function from a vector of simulation results.
"""
function plot_intensity_response(results::Vector; measure::Symbol=:b_wave)
    intensities = Float64[]
    amplitudes = Float64[]

    for r in results
        push!(intensities, r.col.stimulus.I_0)

        trace = r.erg_components[measure]
        if measure == :a_wave
            push!(amplitudes, minimum(trace))  # a-wave is negative
        else
            push!(amplitudes, maximum(trace))  # b-wave, etc. are positive
        end
    end

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
              title="Intensity-Response ($(String(measure)))",
              xlabel="Log Intensity",
              ylabel="Amplitude (a.u.)",
              xscale=log10)
    scatter!(ax, intensities, amplitudes, markersize=10)
    lines!(ax, intensities, amplitudes)

    return fig
end
