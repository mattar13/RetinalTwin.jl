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
    present_types = Symbol[]
    for name in names
        ctype = model.cells[name].cell_type
        if !(ctype in present_types)
            push!(present_types, ctype)
        end
    end
    nrows = length(present_types)

    fig = Figure(size=(1400, max(300, 220 * nrows)))
    v_axes = Axis[]
    ca_axes = Axis[]
    rel_axes = Axis[]

    for (i, ctype) in enumerate(present_types)
        type_names = [nm for nm in names if model.cells[nm].cell_type == ctype]
        ca_key, ca_label = calcium_spec(ctype)
        rel_key, rel_label = release_spec(ctype)
        cols = cgrad(:viridis, max(length(type_names), 2), categorical=true)

        ax_v = Axis(
            fig[i, 1],
            title=i == 1 ? "Voltage" : "",
            subtitle=String(ctype),
            xlabel=i == nrows ? "Time (ms)" : "",
            ylabel="V (mV)",
        )
        for (j, nm) in enumerate(type_names)
            lines!(ax_v, sol.t, state_trace(sol, model, nm, :V), color=cols[j], linewidth=2, label=String(nm))
        end
        vlines!(ax_v, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
        length(type_names) > 1 && axislegend(ax_v, position=:rb)
        i < nrows && hidexdecorations!(ax_v, grid=false)
        push!(v_axes, ax_v)

        ax_ca = Axis(
            fig[i, 2],
            title=i == 1 ? "Calcium" : "",
            subtitle=String(ctype),
            xlabel=i == nrows ? "Time (ms)" : "",
            ylabel=ca_label,
        )
        for (j, nm) in enumerate(type_names)
            lines!(ax_ca, sol.t, state_trace(sol, model, nm, ca_key), color=cols[j], linewidth=2, label=String(nm))
        end
        vlines!(ax_ca, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
        length(type_names) > 1 && axislegend(ax_ca, position=:rb)
        i < nrows && hidexdecorations!(ax_ca, grid=false)
        push!(ca_axes, ax_ca)

        ax_rel = Axis(
            fig[i, 3],
            title=i == 1 ? "Release" : "",
            subtitle=String(ctype),
            xlabel=i == nrows ? "Time (ms)" : "",
            ylabel=rel_label,
        )
        for (j, nm) in enumerate(type_names)
            lines!(ax_rel, sol.t, state_trace(sol, model, nm, rel_key), color=cols[j], linewidth=2, label=String(nm))
        end
        vlines!(ax_rel, [stim_start, stim_end], color=:gray, linestyle=:dash, alpha=0.5)
        length(type_names) > 1 && axislegend(ax_rel, position=:rb)
        i < nrows && hidexdecorations!(ax_rel, grid=false)
        push!(rel_axes, ax_rel)
    end

    linkxaxes!(v_axes...)
    linkxaxes!(ca_axes...)
    linkxaxes!(rel_axes...)

    mkpath(dirname(savepath))
    save(savepath, fig)
    # fig.close()
    return fig
end

function _coords_for_names(
    model::RetinalColumnModel,
    names::Vector{Symbol};
    xbounds::Tuple{Float64,Float64}=(1.0, 1.0),
    ybounds::Tuple{Float64,Float64}=(1.0, 1.0),
)
    n = length(names)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    missing = Int[]
    for (i, nm) in enumerate(names)
        c = model.cells[nm]
        if isfinite(c.x) && isfinite(c.y)
            xs[i] = c.x
            ys[i] = c.y
        else
            xs[i] = NaN
            ys[i] = NaN
            push!(missing, i)
        end
    end

    if !isempty(missing)
        grid = square_grid_coords(length(missing))
        gx = first.(grid)
        gy = last.(grid)
        xlo, xhi = xbounds
        ylo, yhi = ybounds
        gxmax = isempty(gx) ? 1.0 : maximum(gx)
        gymax = isempty(gy) ? 1.0 : maximum(gy)
        xrange = xhi - xlo
        yrange = yhi - ylo
        for (j, idx) in enumerate(missing)
            xs[idx] = gxmax == 1.0 ? (xlo + xhi) / 2 : xlo + xrange * ((gx[j] - 1.0) / (gxmax - 1.0))
            ys[idx] = gymax == 1.0 ? (ylo + yhi) / 2 : ylo + yrange * ((gy[j] - 1.0) / (gymax - 1.0))
        end
    end

    return xs, ys
end

function plot_cell_layout_3d(
    model::RetinalColumnModel;
    savepath::AbstractString,
    z_pc::Real=1.0,
    z_bc::Real=2.0,
    z_gc::Real=3.0,
)
    names = ordered_cells(model)
    pc_names = [nm for nm in names if model.cells[nm].cell_type == :PC]
    onbc_names = [nm for nm in names if model.cells[nm].cell_type == :ONBC]
    offbc_names = [nm for nm in names if model.cells[nm].cell_type == :OFFBC]
    gc_names = [nm for nm in names if model.cells[nm].cell_type == :GC]

    pc_x, pc_y = _coords_for_names(model, pc_names)
    xlo = isempty(pc_x) ? 1.0 : minimum(pc_x)
    xhi = isempty(pc_x) ? 1.0 : maximum(pc_x)
    ylo = isempty(pc_y) ? 1.0 : minimum(pc_y)
    yhi = isempty(pc_y) ? 1.0 : maximum(pc_y)

    bc_names = vcat(onbc_names, offbc_names)
    bc_x, bc_y = _coords_for_names(model, bc_names; xbounds=(xlo, xhi), ybounds=(ylo, yhi))
    n_on = length(onbc_names)
    on_x = bc_x[1:n_on]
    on_y = bc_y[1:n_on]
    off_x = bc_x[(n_on + 1):end]
    off_y = bc_y[(n_on + 1):end]

    gc_x, gc_y = _coords_for_names(model, gc_names; xbounds=(xlo, xhi), ybounds=(ylo, yhi))

    coord_by_name = Dict{Symbol,NTuple{3,Float64}}()
    for (i, nm) in enumerate(pc_names)
        coord_by_name[nm] = (pc_x[i], pc_y[i], Float64(z_pc))
    end
    for (i, nm) in enumerate(onbc_names)
        coord_by_name[nm] = (on_x[i], on_y[i], Float64(z_bc))
    end
    for (i, nm) in enumerate(offbc_names)
        coord_by_name[nm] = (off_x[i], off_y[i], Float64(z_bc))
    end
    for (i, nm) in enumerate(gc_names)
        coord_by_name[nm] = (gc_x[i], gc_y[i], Float64(z_gc))
    end

    fig = Figure(size=(1000, 700))
    ax = Axis3(
        fig[1, 1],
        xlabel="x",
        ylabel="y",
        zlabel="Layer",
        title="Retinal Cell Layout (3D Layers)",
        azimuth=0.8,
        elevation=0.35,
    )

    for (receiver, pres) in sort!(collect(model.connections), by=first)
        haskey(coord_by_name, receiver) || continue
        x1, y1, z1 = coord_by_name[receiver]
        for (pre, _release, w) in pres
            haskey(coord_by_name, pre) || continue
            x0, y0, z0 = coord_by_name[pre]
            lines!(
                ax,
                [x0, x1],
                [y0, y1],
                [z0, z1];
                color=(:gray40, 0.45),
                linewidth=0.8 + 0.4 * clamp(w, 0.0, 3.0),
            )
        end
    end

    if !isempty(pc_x)
        scatter!(ax, pc_x, pc_y, fill(Float64(z_pc), length(pc_x)); markersize=16, color=:goldenrod2, label="Photoreceptors (PC)")
    end
    if !isempty(on_x)
        scatter!(ax, on_x, on_y, fill(Float64(z_bc), length(on_x)); markersize=18, color=:deepskyblue3, marker=:circle, label="ON Bipolar (ONBC)")
    end
    if !isempty(off_x)
        scatter!(ax, off_x, off_y, fill(Float64(z_bc), length(off_x)); markersize=18, color=:orangered3, marker=:utriangle, label="OFF Bipolar (OFFBC)")
    end
    if !isempty(gc_x)
        scatter!(ax, gc_x, gc_y, fill(Float64(z_gc), length(gc_x)); markersize=20, color=:seagreen4, marker=:rect, label="Ganglion (RGC/GC)")
    end

    ax.zticks = ([Float64(z_pc), Float64(z_bc), Float64(z_gc)], ["PC", "BC", "RGC"])
    axislegend(ax, position=:rb)

    mkpath(dirname(savepath))
    save(savepath, fig)
    return fig
end
