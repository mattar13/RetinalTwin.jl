using CairoMakie

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
    names = ordered_cells_by_type(model)
    z_depth = Dict{Symbol,Float64}(k => Float64(v) for (k, v) in RETINAL_LAYER_DEPTH)
    z_depth[:PC] = Float64(z_pc)
    z_depth[:HC] = Float64(z_hc)
    z_depth[:ONBC] = Float64(z_bc)
    z_depth[:OFFBC] = Float64(z_bc)
    z_depth[:MG] = Float64(z_mg)
    z_depth[:A2] = Float64(z_a2)
    z_depth[:GC] = Float64(z_gc)

    present_types_all = present_cell_types(model; names=names)
    present_types = Symbol[ct for ct in RETINAL_CELL_TYPE_ORDER if ct in present_types_all]
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
    names = ordered_cells_by_type(model)

    # Load depth map and compute (z_min, z_max) per cell type
    depth_rows = load_erg_depth_map()
    function _z_range(ct::Symbol)
        label = _cell_type_label(ct)
        zs = [r.z for r in depth_rows if r.cell_type == label]
        if isempty(zs)
            z_fallback = Float64(get(RETINAL_LAYER_DEPTH, ct, 0.0))
            return (z_fallback, z_fallback)
        end
        return (minimum(zs), maximum(zs))
    end

    present_types_all = present_cell_types(model; names=names)
    present_types = Symbol[ct for ct in RETINAL_CELL_TYPE_ORDER if ct in present_types_all]
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

    z_range_by_type = Dict{Symbol,Tuple{Float64,Float64}}()
    for ct in present_types
        z_range_by_type[ct] = _z_range(ct)
    end

    # Use z_mid for connection lines and scatter markers
    coord_by_name = Dict{Symbol,NTuple{3,Float64}}()
    for ct in present_types
        xs, ys = coords_by_type[ct]
        z_lo, z_hi = z_range_by_type[ct]
        z_mid = (z_lo + z_hi) / 2
        for (i, nm) in enumerate(names_by_type[ct])
            coord_by_name[nm] = (xs[i], ys[i], z_mid)
        end
    end

    fig = Figure(size=(1000, 700))
    ax = Axis3(
        fig[1, 1],
        xlabel="x",
        ylabel="y",
        zlabel="Depth",
        title="Retinal Cell Layout (3D Layers)",
        azimuth=0.8,
        elevation=0.35,
    )

    if stimulus_func !== nothing
        ngrid = max(2, Int(stimulus_grid_n))
        xs_grid = collect(range(xlo, xhi; length=ngrid))
        ys_grid = collect(range(ylo, yhi; length=ngrid))
        z_plane = fill(Float64(z_stim), ngrid, ngrid)
        stim_vals = Matrix{Float64}(undef, ngrid, ngrid)
        for i in 1:ngrid, j in 1:ngrid
            stim_vals[i, j] = Float64(stimulus_func(Float64(stimulus_time), xs_grid[i], ys_grid[j]))
        end
        surface!(
            ax,
            xs_grid,
            ys_grid,
            z_plane;
            color=stim_vals,
            alpha=0.5,
            colormap=:inferno,
            shading=NoShading,
            label="Stimulus",
        )
    end

    for (receiver, pres) in sort!(collect(model.connections), by=first)
        haskey(coord_by_name, receiver) || continue
        x1, y1, z1 = coord_by_name[receiver]
        for (pre, _, w) in pres
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

    # Draw vertical depth spans for each cell (x,y from CellRef; z range from depth map)
    for ct in present_types
        xs, ys = coords_by_type[ct]
        isempty(xs) && continue
        z_lo, z_hi = z_range_by_type[ct]
        abs(z_hi - z_lo) < 1e-6 && continue
        st = get(style_by_type, ct, (color=:black, marker=:circle, markersize=14, label=String(ct)))
        n = length(xs)
        seg_xs = Vector{Float64}(undef, 2n)
        seg_ys = Vector{Float64}(undef, 2n)
        seg_zs = Vector{Float64}(undef, 2n)
        for i in 1:n
            seg_xs[2i-1] = xs[i]; seg_xs[2i] = xs[i]
            seg_ys[2i-1] = ys[i]; seg_ys[2i] = ys[i]
            seg_zs[2i-1] = z_lo;  seg_zs[2i] = z_hi
        end
        linesegments!(ax, seg_xs, seg_ys, seg_zs; color=(st.color, 0.5), linewidth=4.0)
    end

    # Scatter markers at z_mid for each cell type
    for ct in present_types
        xs, ys = coords_by_type[ct]
        isempty(xs) && continue
        st = get(style_by_type, ct, (color=:black, marker=:circle, markersize=14, label=String(ct)))
        z_lo, z_hi = z_range_by_type[ct]
        z_mid = (z_lo + z_hi) / 2
        scatter!(
            ax,
            xs,
            ys,
            fill(z_mid, length(xs));
            markersize=st.markersize,
            color=st.color,
            marker=st.marker,
            label=st.label,
        )
    end

    ztick_vals = Float64[]
    ztick_labels = String[]
    for ct in present_types
        z_lo, z_hi = z_range_by_type[ct]
        z_mid = (z_lo + z_hi) / 2
        if !(z_mid in ztick_vals)
            push!(ztick_vals, z_mid)
            push!(ztick_labels, String(ct))
        end
    end
    ax.zticks = (ztick_vals, ztick_labels)
    axislegend(ax, position=:rb)

    mkpath(dirname(savepath))
    save(savepath, fig)
    return fig
end
