using CairoMakie

const Z_DEPTH = Dict(
    :PC => 1.0,
    :HC => 1.5,
    :ONBC => 2.0,
    :OFFBC => 2.0,
    :MG => 2.3,
    :A2 => 2.6,
    :GC => 3.0,
)

const CELL_TYPE_ORDER = sort(collect(keys(Z_DEPTH)), by=ct -> (Z_DEPTH[ct], String(ct)))

function _cell_sort_key(name::Symbol, cell::CellRef)
    s = String(name)
    m = match(r"\d+$", s)
    idx = m === nothing ? 0 : parse(Int, m.match)
    rank = findfirst(==(cell.cell_type), CELL_TYPE_ORDER)
    return (rank === nothing ? 99 : rank, idx, s)
end

ordered_cells(model::RetinalColumnModel) =
    sort!(collect(keys(model.cells)), by=name -> _cell_sort_key(name, model.cells[name]))

function present_cell_types(model::RetinalColumnModel)
    types = Symbol[]
    for nm in ordered_cells(model)
        ct = model.cells[nm].cell_type
        if !(ct in types)
            push!(types, ct)
        end
    end
    return types
end

function calcium_spec(cell_type::Symbol)
    if cell_type == :PC
        return (:Ca_f, "Ca (Ca_f)")
    elseif cell_type == :HC
        return (:c, "Ca (c)")
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
    elseif cell_type == :HC
        return (:I, "Release proxy (I)")
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
    z_pc::Real=1.0,
    z_hc::Real=1.5,
    z_bc::Real=2.0,
    z_mg::Real=2.3,
    z_a2::Real=2.6,
    z_gc::Real=3.0,
)
    names = ordered_cells(model)
    z_depth = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in Z_DEPTH)
    z_depth[:PC] = Float64(z_pc)
    z_depth[:HC] = Float64(z_hc)
    z_depth[:ONBC] = Float64(z_bc)
    z_depth[:OFFBC] = Float64(z_bc)
    z_depth[:MG] = Float64(z_mg)
    z_depth[:A2] = Float64(z_a2)
    z_depth[:GC] = Float64(z_gc)
    present_types_all = present_cell_types(model)
    present_types = Symbol[ct for ct in CELL_TYPE_ORDER if ct in present_types_all]
    for ct in present_types_all
        if !(ct in present_types)
            push!(present_types, ct)
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
    stimulus_func=nothing,
    stimulus_time::Real=0.0,
    stimulus_grid_n::Int=41,
    z_stim::Real=0.0,
)
    names = ordered_cells(model)
    z_depth = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in Z_DEPTH)
    present_types_all = present_cell_types(model)
    present_types = Symbol[ct for ct in CELL_TYPE_ORDER if ct in present_types_all]
    for ct in present_types_all
        if !(ct in present_types)
            push!(present_types, ct)
        end
    end

    style_by_type = Dict(
        :PC => (color=:goldenrod2, marker=:circle, markersize=16, label="Photoreceptors (PC)"),
        :HC => (color=:orchid3, marker=:diamond, markersize=16, label="Horizontal (HC)"),
        :ONBC => (color=:deepskyblue3, marker=:circle, markersize=18, label="ON Bipolar (ONBC)"),
        :OFFBC => (color=:orangered3, marker=:utriangle, markersize=18, label="OFF Bipolar (OFFBC)"),
        :MG => (color=:slategray3, marker=:cross, markersize=16, label="Muller (MG)"),
        :A2 => (color=:mediumpurple3, marker=:hexagon, markersize=18, label="A2 Amacrine (A2)"),
        :GC => (color=:seagreen4, marker=:rect, markersize=20, label="Ganglion (RGC/GC)"),
    )

    names_by_type = Dict{Symbol,Vector{Symbol}}()
    for ct in present_types
        names_by_type[ct] = [nm for nm in names if model.cells[nm].cell_type == ct]
    end

    pc_names = get(names_by_type, :PC, Symbol[])

    pc_x, pc_y = _coords_for_names(model, pc_names)
    xlo = isempty(pc_x) ? 1.0 : minimum(pc_x)
    xhi = isempty(pc_x) ? 1.0 : maximum(pc_x)
    ylo = isempty(pc_y) ? 1.0 : minimum(pc_y)
    yhi = isempty(pc_y) ? 1.0 : maximum(pc_y)
    xlo == xhi && (xlo -= 0.5; xhi += 0.5)
    ylo == yhi && (ylo -= 0.5; yhi += 0.5)

    coords_by_type = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
    for ct in present_types
        coords_by_type[ct] = _coords_for_names(model, names_by_type[ct]; xbounds=(xlo, xhi), ybounds=(ylo, yhi))
    end

    coord_by_name = Dict{Symbol,NTuple{3,Float64}}()
    for ct in present_types
        xs, ys = coords_by_type[ct]
        z = Float64(get(z_depth, ct, 0.0))
        for (i, nm) in enumerate(names_by_type[ct])
            coord_by_name[nm] = (xs[i], ys[i], z)
        end
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

    if stimulus_func !== nothing
        ngrid = max(2, Int(stimulus_grid_n))
        xs = collect(range(xlo, xhi; length=ngrid))
        ys = collect(range(ylo, yhi; length=ngrid))
        z_plane = fill(Float64(z_stim), ngrid, ngrid)
        stim_vals = Matrix{Float64}(undef, ngrid, ngrid)
        for i in 1:ngrid, j in 1:ngrid
            stim_vals[i, j] = Float64(stimulus_func(Float64(stimulus_time), xs[i], ys[j]))
        end
        surface!(
            ax,
            xs,
            ys,
            z_plane;
            color=stim_vals,
            alpha = 0.5,
            colormap=:inferno,
            shading=NoShading,
            label = "Stimulus",
        )
    end

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

    for ct in present_types
        xs, ys = coords_by_type[ct]
        isempty(xs) && continue
        st = get(style_by_type, ct, (color=:black, marker=:circle, markersize=14, label=String(ct)))
        z = Float64(get(z_depth, ct, 0.0))
        scatter!(
            ax,
            xs,
            ys,
            fill(z, length(xs));
            markersize=st.markersize,
            color=st.color,
            marker=st.marker,
            label=st.label,
        )
    end

    ztick_vals = Float64[]
    ztick_labels = String[]
    for ct in present_types
        z = Float64(get(z_depth, ct, 0.0))
        if !(z in ztick_vals)
            push!(ztick_vals, z)
            push!(ztick_labels, String(ct))
        end
    end
    ax.zticks = (ztick_vals, ztick_labels)
    axislegend(ax, position=:rb)

    mkpath(dirname(savepath))
    save(savepath, fig)
    return fig
end
