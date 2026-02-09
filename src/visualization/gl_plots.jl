# ============================================================
# gl_plots.jl — GLMakie plotting utilities
# ============================================================

using GLMakie

"""
    plot_erg_gl(result; show_components=true, time_range=nothing)

Plot the total ERG and optionally decomposed components with GLMakie.
`result` is the NamedTuple from `simulate_flash`.
"""
function plot_erg_gl(result; show_components::Bool=true, time_range=nothing)
    t = result.t
    erg = result.erg
    comps = result.erg_components

    if time_range !== nothing
        mask = (t .>= time_range[1]) .& (t .<= time_range[2])
        t = t[mask]
        erg = erg[mask]
        comps = Dict(k => v[mask] for (k, v) in comps)
    end

    if show_components
        fig = Figure(size=(900, 800))

        ax1 = Axis(fig[1, 1], title="Total ERG", xlabel="Time (ms)", ylabel="Amplitude (a.u.)")
        lines!(ax1, t, erg, color=:black, linewidth=2)
        hlines!(ax1, [0.0], color=:gray, linestyle=:dash)

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
    plot_cell_voltages_gl(result; cell_types=[:rod, :on_bc, :a2, :gc], time_range=nothing)

Plot membrane potential traces for selected cell types with GLMakie.
"""
function plot_cell_voltages_gl(result; cell_types::Vector{Symbol}=[:rod, :on_bc, :a2, :gc],
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
            for ci in 1:min(n_cells, 5)
                lines!(ax, t, V[:, ci], linewidth=1)
            end
            if n_cells > 5
                lines!(ax, t, vec(mean(V, dims=2)), color=:black, linewidth=2, linestyle=:dash)
            end
        end
    end

    return fig
end

"""
    plot_phototransduction_breakdown(result; cell=:rod, cell_index=1, time_range=nothing)

Plot a breakdown of phototransduction state variables for one photoreceptor.
"""
function plot_phototransduction_breakdown(result; cell::Symbol=:rod, cell_index::Int=1, time_range=nothing)
    t = result.t
    sol = result.solution
    stimulus = [compute_stimulus(result.col.stimulus, ti) for ti in t]

    if cell == :rod
        base = result.sidx.rod[1] + (cell_index - 1) * 6
    elseif cell == :cone
        base = result.sidx.cone[1] + (cell_index - 1) * 6
    else
        error("Phototransduction breakdown only supports :rod or :cone, got $(cell).")
    end

    r_star = [u[base] for u in sol.u]
    g_state = [u[base + 1] for u in sol.u]
    ca = [u[base + 2] for u in sol.u]
    v = [u[base + 3] for u in sol.u]
    h = [u[base + 4] for u in sol.u]
    glu = [u[base + 5] for u in sol.u]

    if time_range !== nothing
        mask = (t .>= time_range[1]) .& (t .<= time_range[2])
        t = t[mask]
        stimulus = stimulus[mask]
        r_star = r_star[mask]
        g_state = g_state[mask]
        ca = ca[mask]
        v = v[mask]
        h = h[mask]
        glu = glu[mask]
    end

    fig = Figure(size=(950, 900))
    axes = Axis[]

    push!(axes, Axis(fig[1, 1], title="Phototransduction breakdown ($(uppercase(String(cell))) $(cell_index))",
                     xlabel="Time (ms)", ylabel="Stimulus (ph/μm²/ms)"))
    lines!(axes[end], t, stimulus, color=:orange, linewidth=2)

    push!(axes, Axis(fig[2, 1], xlabel="Time (ms)", ylabel="R* (a.u.)"))
    lines!(axes[end], t, r_star, color=:blue, linewidth=2)

    push!(axes, Axis(fig[3, 1], xlabel="Time (ms)", ylabel="G (a.u.)"))
    lines!(axes[end], t, g_state, color=:purple, linewidth=2)

    push!(axes, Axis(fig[4, 1], xlabel="Time (ms)", ylabel="Ca (a.u.)"))
    lines!(axes[end], t, ca, color=:teal, linewidth=2)

    push!(axes, Axis(fig[5, 1], xlabel="Time (ms)", ylabel="V (mV)"))
    lines!(axes[end], t, v, color=:black, linewidth=2)

    push!(axes, Axis(fig[6, 1], xlabel="Time (ms)", ylabel="Glu (a.u.)"))
    lines!(axes[end], t, glu, color=:red, linewidth=2)

    linkxaxes!(axes...)

    return fig
end
