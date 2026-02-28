#We are working on a more complex mapping model that will add some convienance
struct CellRef
    name::Symbol
    cell_type::Symbol
    offset::Int
    nstate::Int
    outidx::NamedTuple
    x::Float64
    y::Float64
end

function CellRef(cell_type::Symbol, num::Int, offset::Int; x::Real=NaN, y::Real=NaN)
    symbol = Symbol(cell_type, num)
    if cell_type == :PC
        return CellRef(symbol, cell_type, offset, n_PC_STATES, PC_IC_MAP, Float64(x), Float64(y))
    elseif cell_type == :HC
        return CellRef(symbol, cell_type, offset, n_HC_STATES, HC_IC_MAP, Float64(x), Float64(y))
    elseif cell_type == :ONBC
        return CellRef(symbol, cell_type, offset, n_ONBC_STATES, ONBC_IC_MAP, Float64(x), Float64(y))
    elseif cell_type == :OFFBC
        return CellRef(symbol, cell_type, offset, n_OFFBC_STATES, OFFBC_IC_MAP, Float64(x), Float64(y))
    elseif cell_type == :A2
        return CellRef(symbol, cell_type, offset, n_A2_STATES, A2_IC_MAP, Float64(x), Float64(y))
    elseif cell_type == :GC
        return CellRef(symbol, cell_type, offset, n_GC_STATES, GC_IC_MAP, Float64(x), Float64(y))
    elseif cell_type == :MG
        return CellRef(symbol, cell_type, offset, n_MG_STATES, MG_IC_MAP, Float64(x), Float64(y))
    else
        error("Invalid cell type: $cell_type")
    end
end

global_idx(cell::CellRef, key::Symbol) = cell.offset + cell.outidx[key] - 1

(cell::CellRef)(key::Symbol) = global_idx(cell, key)

function display_global_idxs(cell::CellRef)
    for i in 1:cell.nstate
        key = keys(cell.outidx)[i]
        val = global_idx(cell, key)
        println("Global index for $(cell.name) $(key) is $(val)")
    end
    println("--------------------------------")
end
"""
    RetinalColumnModel
"""
struct RetinalColumnModel
    cells::Dict{Symbol,CellRef}
    connections::Dict{Symbol,Vector{Tuple{Symbol,Symbol,Float64}}}
end

"""
    save_mapping(path, model, u0)

Save `model` and `u0` to a JSON file.
The JSON has top-level keys: `schema_version`, `cells`, `connections`, and `u0`.
"""
function save_mapping(path::AbstractString, model::RetinalColumnModel, u0::AbstractVector{<:Real})
    json_num_or_null(x::Float64) = isfinite(x) ? x : nothing

    cell_entries = [
        Dict(
            "name" => String(cell.name),
            "cell_type" => String(cell.cell_type),
            "offset" => cell.offset,
            "x" => json_num_or_null(cell.x),
            "y" => json_num_or_null(cell.y),
        ) for cell in sort!(collect(values(model.cells)), by=cell -> cell.offset)
    ]

    connection_entries = Dict{String,Any}()
    for (receiver, pres) in model.connections
        connection_entries[String(receiver)] = [
            Dict(
                "pre" => String(pre),
                "release" => String(release),
                "w" => w,
            ) for (pre, release, w) in pres
        ]
    end

    payload = Dict(
        "schema_version" => 1,
        "cells" => cell_entries,
        "connections" => connection_entries,
        "u0" => Float64.(u0),
    )

    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, payload, 2)
    end
    return path
end

"""
    save_mapping(path, model)

Save only `model` with an empty `u0`.
"""
function save_mapping(path::AbstractString, model::RetinalColumnModel)
    return save_mapping(path, model, Float64[])
end

function _cell_layout(cell_type::Symbol)
    if cell_type == :PC
        return n_PC_STATES, PC_IC_MAP
    elseif cell_type == :HC
        return n_HC_STATES, HC_IC_MAP
    elseif cell_type == :ONBC
        return n_ONBC_STATES, ONBC_IC_MAP
    elseif cell_type == :OFFBC
        return n_OFFBC_STATES, OFFBC_IC_MAP
    elseif cell_type == :A2
        return n_A2_STATES, A2_IC_MAP
    elseif cell_type == :GC
        return n_GC_STATES, GC_IC_MAP
    elseif cell_type == :MG
        return n_MG_STATES, MG_IC_MAP
    else
        error("Invalid cell type in mapping file: $cell_type")
    end
end

"""
    load_mapping(path)

Load a mapping JSON saved by `save_mapping`.
Returns `(model, u0)`.
"""
function load_mapping(path::AbstractString)::Tuple{RetinalColumnModel,Vector{Float64}}
    payload = JSON.parsefile(path)

    cells = Dict{Symbol,CellRef}()
    for entry in payload["cells"]
        name = Symbol(entry["name"])
        cell_type = Symbol(entry["cell_type"])
        offset = Int(entry["offset"])
        x = isnothing(entry["x"]) ? NaN : Float64(entry["x"])
        y = isnothing(entry["y"]) ? NaN : Float64(entry["y"])
        nstate, outidx = _cell_layout(cell_type)
        cells[name] = CellRef(name, cell_type, offset, nstate, outidx, x, y)
    end

    connections = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Float64}}}()
    for (receiver, pres) in payload["connections"]
        entries = Tuple{Symbol,Symbol,Float64}[]
        for edge in pres
            push!(entries, (Symbol(edge["pre"]), Symbol(edge["release"]), Float64(edge["w"])))
        end
        connections[Symbol(receiver)] = entries
    end

    u0 = haskey(payload, "u0") ? Float64.(payload["u0"]) : Float64[]
    model = RetinalColumnModel(cells, connections)
    return model, u0
end

cell_range(cell::CellRef)::UnitRange{Int} = cell.offset:(cell.offset + cell.nstate - 1)

uview(u, cell::CellRef) = @view u[cell_range(cell)]

duview(du, cell::CellRef) = @view du[cell_range(cell)]

get_out(u, cell::CellRef, key::Symbol) = u[cell(key)]

function connect!(model::RetinalColumnModel, receiver::Symbol, pre::Symbol; release::Symbol=:Glu, w::Real=1.0)
    haskey(model.cells, receiver) || error("Unknown receiver cell: $receiver")
    haskey(model.cells, pre) || error("Unknown presynaptic cell: $pre")
    if !haskey(model.connections, receiver)
        model.connections[receiver] = Tuple{Symbol,Symbol,Float64}[]
    end
    push!(model.connections[receiver], (pre, release, Float64(w)))
    return model
end

function connect!(model::RetinalColumnModel, receiver::Symbol, pres::Vector{Symbol}; release::Symbol=:Glu, w::Real=1.0)
    for pre in pres
        connect!(model, receiver, pre; release=release, w=w)
    end
    return model
end