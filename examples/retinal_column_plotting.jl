using CairoMakie

const CELL_TYPE_ORDER = Dict(
    :PC => 1,
    :ONBC => 2,
    :OFFBC => 3,
    :A2 => 4,
    :GC => 5,
    :MG => 6,
)

function _cell_sort_key(name::Symbol, cell::CellRef)
    s = String(name)
    m = match(r"\d+$", s)
    idx = m === nothing ? 0 : parse(Int, m.match)
    return (get(CELL_TYPE_ORDER, cell.cell_type, 99), idx, s)
end

ordered_cells(model::RetinalColumnModel) =
    sort!(collect(keys(model.cells)), by=name -> _cell_sort_key(name, model.cells[name]))

function calcium_spec(cell_type::Symbol)
    if cell_type == :PC
        return (:Ca_f, "Ca (Ca_f)")
    elseif cell_type == :ONBC || cell_type == :OFFBC || cell_type == :A2
        return (:c, "Ca (c)")
    elseif cell_type == :GC
        return (:sE, "Ca proxy (sE)")
    elseif cell_type == :MG
        return (:K_o_end, "Ca proxy (K_o_end)")
    else
        error("Unsupported cell type: $cell_type")
    end
end

function release_spec(cell_type::Symbol)
    if cell_type == :PC || cell_type == :ONBC || cell_type == :OFFBC
        return (:Glu, "Glutamate")
    elseif cell_type == :A2
        return (:Y, "Release proxy (Y)")
    elseif cell_type == :GC
        return (:sE, "Release proxy (sE)")
    elseif cell_type == :MG
        return (:Glu_o, "Glu pool (Glu_o)")
    else
        error("Unsupported cell type: $cell_type")
    end
end

state_trace(sol, model::RetinalColumnModel, name::Symbol, key::Symbol) =
    [get_out(ui, model.cells[name], key) for ui in sol.u]

function plot_all_cells_v_ca_release(
    sol,
    model::RetinalColumnModel;
    stim_start::Real,
    stim_end::Real,
    savepath::AbstractString,
)
    names = ordered_cells(model)
    nrows = length(names)

    fig = Figure(size=(1400, max(300, 220 * nrows)))
    v_axes = Axis[]
    ca_axes = Axis[]
    rel_axes = Axis[]

    for (i, name) in enumerate(names)
        cell = model.cells[name]
        ca_key, ca_label = calcium_spec(cell.cell_type)
        rel_key, rel_label = release_spec(cell.cell_type)

        ax_v = Axis(
            fig[i, 1],
            title=i == 1 ? "Voltage" : "",
            subtitle=String(name),
            xlabel=i == nrows ? "Time (ms)" : "",
            ylabel="V (mV)",
        )
        lines!(ax_v, sol.t, state_trace(sol, model, name, :V), color=:blue, linewidth=2)
        vlines!(ax_v, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
        i < nrows && hidexdecorations!(ax_v, grid=false)
        push!(v_axes, ax_v)

        ax_ca = Axis(
            fig[i, 2],
            title=i == 1 ? "Calcium" : "",
            subtitle=String(name),
            xlabel=i == nrows ? "Time (ms)" : "",
            ylabel=ca_label,
        )
        lines!(ax_ca, sol.t, state_trace(sol, model, name, ca_key), color=:purple, linewidth=2)
        vlines!(ax_ca, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
        i < nrows && hidexdecorations!(ax_ca, grid=false)
        push!(ca_axes, ax_ca)

        ax_rel = Axis(
            fig[i, 3],
            title=i == 1 ? "Release" : "",
            subtitle=String(name),
            xlabel=i == nrows ? "Time (ms)" : "",
            ylabel=rel_label,
        )
        lines!(ax_rel, sol.t, state_trace(sol, model, name, rel_key), color=:green, linewidth=2)
        vlines!(ax_rel, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
        i < nrows && hidexdecorations!(ax_rel, grid=false)
        push!(rel_axes, ax_rel)
    end

    linkxaxes!(v_axes...)
    linkxaxes!(ca_axes...)
    linkxaxes!(rel_axes...)

    mkpath(dirname(savepath))
    save(savepath, fig)
    return fig
end
