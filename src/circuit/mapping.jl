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

function square_grid_coords(n::Int)
    n < 0 && error("n must be non-negative, got $n")
    nx = max(1, ceil(Int, sqrt(n)))
    coords = Tuple{Float64,Float64}[]
    sizehint!(coords, n)
    for i in 1:n
        col = mod(i - 1, nx)
        row = div(i - 1, nx)
        push!(coords, (col + 1.0, row + 1.0))
    end
    return coords
end

function build_column(;nPC::Int=16, nHC::Int=4, nONBC::Int=4, nOFFBC::Int=0, nA2::Int=4, nGC::Int=1, nMG::Int=4, onbc_pool_size::Int=4, pc_coords=nothing)
    if pc_coords !== nothing && length(pc_coords) != nPC
        error("pc_coords length ($(length(pc_coords))) must equal nPC ($nPC)")
    end
    onbc_pool_size < 1 && error("onbc_pool_size must be >= 1, got $onbc_pool_size")
    nPC > nONBC * onbc_pool_size && error(
        "Insufficient ONBC pooling capacity: nPC=$nPC, nONBC=$nONBC, onbc_pool_size=$onbc_pool_size. " *
        "Increase nONBC or onbc_pool_size."
    )

    params = default_retinal_params()
    cells = Dict{Symbol,CellRef}()
    connections = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Float64}}}()
    u0_parts = Vector{Vector{Float64}}()

    offset = 1
    for i in 1:nPC
        x, y = pc_coords === nothing ? (Float64(i), 1.0) : (Float64(pc_coords[i][1]), Float64(pc_coords[i][2]))
        cell = CellRef(:PC, i, offset; x=x, y=y)
        cells[cell.name] = cell
        push!(u0_parts, photoreceptor_state(params.PHOTORECEPTOR_PARAMS))
        offset += cell.nstate
    end

    for i in 1:nHC
        cell = CellRef(:HC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, horizontal_state(params.HORIZONTAL_PARAMS))
        offset += cell.nstate
    end

    for i in 1:nONBC
        cell = CellRef(:ONBC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, on_bipolar_state(params.ON_BIPOLAR_PARAMS))
        offset += cell.nstate
    end

    for i in 1:nOFFBC
        cell = CellRef(:OFFBC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, off_bipolar_state(params.OFF_BIPOLAR_PARAMS))
        offset += cell.nstate
    end

    for i in 1:nA2
        cell = CellRef(:A2, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, a2_amacrine_state(params.A2_AMACRINE_PARAMS))
        offset += cell.nstate
    end

    for i in 1:nGC
        cell = CellRef(:GC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, ganglion_state(params.GANGLION_PARAMS))
        offset += cell.nstate
    end

    for i in 1:nMG
        cell = CellRef(:MG, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, muller_state(params.MULLER_PARAMS))
        offset += cell.nstate
    end

    model = RetinalColumnModel(cells, connections)

    pc_names = Symbol[Symbol(:PC, i) for i in 1:nPC]

    for i in 1:nHC
        hc = Symbol(:HC, i)
        connect!(model, hc, pc_names; release=:Glu, w=1.0)
    end

    for i in 1:nHC, j in 1:nHC
        if i != j
            connect!(model, Symbol(:HC, i), Symbol(:HC, j); release=:V, w=1.0)
        end
    end

    for i in 1:nONBC
        onbc = Symbol(:ONBC, i)
        first_pc = (i - 1) * onbc_pool_size + 1
        last_pc = min(i * onbc_pool_size, nPC)
        if first_pc <= nPC
            connect!(model, onbc, pc_names[first_pc:last_pc]; release=:Glu, w=1.0)
        end
    end

    for i in 1:nOFFBC
        offbc = Symbol(:OFFBC, i)
        connect!(model, offbc, pc_names; release=:Glu, w=1.0)
    end

    onbc_names = Symbol[Symbol(:ONBC, i) for i in 1:nONBC]
    for i in 1:nA2
        a2 = Symbol(:A2, i)
        connect!(model, a2, onbc_names; release=:Glu, w=1.0)
        connect!(model, a2, onbc_names; release=:V, w=1.0)
    end

    a2_names = Symbol[Symbol(:A2, i) for i in 1:nA2]
    for i in 1:nONBC
        onbc = Symbol(:ONBC, i)
        connect!(model, onbc, a2_names; release=:V, w=1.0)
    end

    offbc_names = Symbol[Symbol(:OFFBC, i) for i in 1:nOFFBC]
    for i in 1:nGC
        gc = Symbol(:GC, i)
        connect!(model, gc, onbc_names; release=:Glu, w=1.0)
        connect!(model, gc, offbc_names; release=:Glu, w=1.0)
        connect!(model, gc, a2_names; release=:Y, w=1.0)
    end

    u0 = isempty(u0_parts) ? Float64[] : vcat(u0_parts...)
    return model, u0
end
