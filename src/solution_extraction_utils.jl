const RETINAL_LAYER_DEPTH = Dict(
    :PC => 1.0,
    :HC => 1.5,
    :ONBC => 2.0,
    :OFFBC => 2.0,
    :MG => 2.3,
    :A2 => 2.6,
    :GC => 3.0,
)

const RETINAL_CELL_TYPE_ORDER = sort(collect(keys(RETINAL_LAYER_DEPTH)), by=ct -> (RETINAL_LAYER_DEPTH[ct], String(ct)))

@inline function _cell_suffix_index(name::Symbol)
    m = match(r"\d+$", String(name))
    return m === nothing ? 0 : parse(Int, m.match)
end

ordered_cells(model::RetinalColumnModel) =
    sort!(collect(keys(model.cells)), by=nm -> model.cells[nm].offset)

function ordered_cells_by_type(model::RetinalColumnModel; cell_type_order::AbstractVector{Symbol}=RETINAL_CELL_TYPE_ORDER)
    names = ordered_cells(model)
    rank = Dict(ct => i for (i, ct) in enumerate(cell_type_order))
    sort!(names, by=nm -> begin
        ct = model.cells[nm].cell_type
        (get(rank, ct, typemax(Int)), _cell_suffix_index(nm), String(nm))
    end)
end

function present_cell_types(model::RetinalColumnModel; names::AbstractVector{Symbol}=ordered_cells_by_type(model))
    types = Symbol[]
    for nm in names
        ct = model.cells[nm].cell_type
        ct in types || push!(types, ct)
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

function state_trace(sol, model::RetinalColumnModel, name::Symbol, key::Symbol)
    cell = model.cells[name]
    n = length(sol.u)
    vals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        vals[i] = get_out(sol.u[i], cell, key)
    end
    return vals
end

@inline function finite_mean(x)
    s = 0.0
    n = 0
    @inbounds for v in x
        if isfinite(v)
            s += v
            n += 1
        end
    end
    return n == 0 ? NaN : (s / n)
end
