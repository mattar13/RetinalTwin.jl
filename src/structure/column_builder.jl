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
    if nONBC > 0
        onbc_pool_size < 1 && error("onbc_pool_size must be >= 1, got $onbc_pool_size")
        nPC > nONBC * onbc_pool_size && error(
            "Insufficient ONBC pooling capacity: nPC=$nPC, nONBC=$nONBC, onbc_pool_size=$onbc_pool_size. " *
            "Increase nONBC or onbc_pool_size."
        )
    end

    params = load_all_params()
    cells = Dict{Symbol,CellRef}()
    connections = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Float64}}}()
    u0_parts = Vector{Vector{Float64}}()

    offset = 1
    for i in 1:nPC
        x, y = pc_coords === nothing ? (Float64(i), 1.0) : (Float64(pc_coords[i][1]), Float64(pc_coords[i][2]))
        cell = CellRef(:PC, i, offset; x=x, y=y)
        cells[cell.name] = cell
        push!(u0_parts, photoreceptor_state(params.PHOTO))
        offset += cell.nstate
    end

    for i in 1:nHC
        cell = CellRef(:HC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, horizontal_state(params.HC))
        offset += cell.nstate
    end

    for i in 1:nONBC
        cell = CellRef(:ONBC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, on_bipolar_state(params.ONBC))
        offset += cell.nstate
    end

    for i in 1:nOFFBC
        cell = CellRef(:OFFBC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, off_bipolar_state(params.OFFBC))
        offset += cell.nstate
    end

    for i in 1:nA2
        cell = CellRef(:A2, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, a2_amacrine_state(params.A2))
        offset += cell.nstate
    end

    for i in 1:nGC
        cell = CellRef(:GC, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, ganglion_state(params.GC))
        offset += cell.nstate
    end

    for i in 1:nMG
        cell = CellRef(:MG, i, offset)
        cells[cell.name] = cell
        push!(u0_parts, muller_state(params.MULLER))
        offset += cell.nstate
    end

    model = RetinalColumnModel(cells, connections)

    # HC ← PC (glutamate); HC ↔ HC gap junctions
    nHC > 0 && nPC > 0  && connect_populations!(model, :PC,   :HC;   release=:Glu)
    nHC > 1             && connect_populations!(model, :HC,   :HC;   release=:V)

    # ONBC ← PC: sequential block pooling (index-based, not spatial)
    for i in 1:nONBC
        first_pc = (i - 1) * onbc_pool_size + 1
        last_pc  = min(i * onbc_pool_size, nPC)
        first_pc > nPC && continue
        connect!(model, Symbol(:ONBC, i), [Symbol(:PC, j) for j in first_pc:last_pc]; release=:Glu)
    end

    # OFFBC ← PC (glutamate, all-to-all)
    nOFFBC > 0 && nPC > 0   && connect_populations!(model, :PC,   :OFFBC; release=:Glu)

    # A2 ← ONBC (glutamate + gap junction); ONBC ← A2 (gap junction)
    nA2 > 0 && nONBC > 0    && connect_populations!(model, :ONBC, :A2;   release=:Glu)
    nA2 > 0 && nONBC > 0    && connect_populations!(model, :ONBC, :A2;   release=:V)
    nONBC > 0 && nA2 > 0    && connect_populations!(model, :A2,   :ONBC; release=:V)

    # GC ← ONBC (Glu), OFFBC (Glu), A2 (glycine/Y)
    nGC > 0 && nONBC > 0    && connect_populations!(model, :ONBC,  :GC;  release=:Glu)
    nGC > 0 && nOFFBC > 0   && connect_populations!(model, :OFFBC, :GC;  release=:Glu)
    nGC > 0 && nA2 > 0      && connect_populations!(model, :A2,    :GC;  release=:Y)

    u0 = isempty(u0_parts) ? Float64[] : vcat(u0_parts...)
    return model, u0
end

function build_photoreceptor_column(nPC::Int=16; nHC::Int=4, pc_coords=nothing)
    return build_column(;
        nPC=nPC,
        nHC=nHC,
        nONBC=0,
        onbc_pool_size=1,
        nOFFBC=0,
        nA2=0,
        nGC=0,
        nMG=0,
        pc_coords=pc_coords,
    )
end

function _default_column_json_path()
    return normpath(joinpath(@__DIR__, "..", "..", "examples", "data", "default_column.json"))
end

function default_build_column(; rebuild::Bool=false)
    save_path = _default_column_json_path()
    if !rebuild && isfile(save_path)
        return load_mapping(save_path)
    end
    pc_coords = square_grid_coords(16)
    model, u0 = build_column(;
        nPC=16, nHC=4, nONBC=4, onbc_pool_size=4,
        nOFFBC=4, nA2=4, nGC=1, nMG=4,
        pc_coords=pc_coords,
    )
    mkpath(dirname(save_path))
    save_mapping(save_path, model, u0)
    return model, u0
end

# ─── Position helpers ──────────────────────────────────────────────────────────

function _set_cell_position!(model::RetinalColumnModel, name::Symbol, x::Float64, y::Float64)
    old = model.cells[name]
    model.cells[name] = CellRef(old.name, old.cell_type, old.offset, old.nstate, old.outidx, x, y)
end

_cell_distance(a::CellRef, b::CellRef) = sqrt((a.x - b.x)^2 + (a.y - b.y)^2)

function _cells_of_type(model::RetinalColumnModel, cell_type::Symbol)
    return sort(
        [c for c in values(model.cells) if c.cell_type == cell_type],
        by = c -> c.name,
    )
end

# ─── Alignment functions ────────────────────────────────────────────────────────

function align_grid!(model::RetinalColumnModel, cell_type::Symbol)
    cells = _cells_of_type(model, cell_type)
    n = length(cells)
    n == 0 && return model
    nx = max(1, ceil(Int, sqrt(n)))
    for (i, cell) in enumerate(cells)
        col = mod(i - 1, nx)
        row = div(i - 1, nx)
        _set_cell_position!(model, cell.name, col + 1.0, row + 1.0)
    end
    return model
end

function align_circle!(model::RetinalColumnModel, cell_type::Symbol; radius::Float64=1.0)
    cells = _cells_of_type(model, cell_type)
    n = length(cells)
    n == 0 && return model
    for (i, cell) in enumerate(cells)
        θ = 2π * (i - 1) / n
        _set_cell_position!(model, cell.name, radius * cos(θ), radius * sin(θ))
    end
    return model
end

function align_random!(
    model::RetinalColumnModel,
    cell_type::Symbol;
    x_coords::Tuple{Float64,Float64} = (0.0, 1.0),
    y_coords::Tuple{Float64,Float64} = (0.0, 1.0),
    distribution = Uniform(),
)
    cells = _cells_of_type(model, cell_type)
    n = length(cells)
    n == 0 && return model
    for cell in cells
        x = clamp(rand(distribution), x_coords[1], x_coords[2])
        y = clamp(rand(distribution), y_coords[1], y_coords[2])
        _set_cell_position!(model, cell.name, x, y)
    end
    return model
end

# ─── Population-level connection helpers ───────────────────────────────────────

"""
    connect_populations!(model, sender_type, receiver_type; release, w)

Connect every cell in `sender_type` to every cell in `receiver_type`.
Gap junctions are expressed by passing the same type for both arguments
(e.g. `connect_populations!(model, :HC, :HC; release=:V)`).
"""
function connect_populations!(
    model::RetinalColumnModel,
    sender_type::Symbol,
    receiver_type::Symbol;
    release::Symbol = :Glu,
    w::Float64 = 1.0,
    skip_self::Bool = true,
)
    senders   = _cells_of_type(model, sender_type)
    receivers = _cells_of_type(model, receiver_type)
    isempty(senders)   && error("No cells of type $sender_type in model")
    isempty(receivers) && error("No cells of type $receiver_type in model")
    for rcell in receivers
        for scell in senders
            skip_self && rcell.name == scell.name && continue
            connect!(model, rcell.name, scell.name; release=release, w=w)
        end
    end
    return model
end

"""
    pool_connections!(model, sender_syms, receiver_type; release, w, mode, k)

Rewire a specific set of sender cells to all cells in `receiver_type`.
Existing connections from `sender_syms` on each receiver are removed first
(surgical: all other connections are preserved).

**Modes**
- `:nearest_neighbors` — each receiver connects to its `k` nearest senders (k::Int)
- `:distance_inclusive` — each receiver connects to all senders within distance `k`
- `:distance_exclusive` — each receiver connects to at most one sender (its nearest
  within `k`); a sender may serve multiple receivers
"""
function pool_connections!(
    model::RetinalColumnModel,
    sender_syms::Vector{Symbol},
    receiver_type::Symbol;
    release::Symbol = :Glu,
    w::Float64 = 1.0,
    mode::Symbol = :nearest_neighbors,
    k = 1,
)
    for s in sender_syms
        haskey(model.cells, s) || error("Unknown sender cell: $s")
    end
    sender_cells = [model.cells[s] for s in sender_syms]
    sender_set   = Set(sender_syms)

    receivers = _cells_of_type(model, receiver_type)
    isempty(receivers) && error("No cells of type $receiver_type in model")

    # Surgical removal: strip existing connections from sender_syms on each receiver
    for rcell in receivers
        rname = rcell.name
        if haskey(model.connections, rname)
            filter!(conn -> !(conn[1] in sender_set), model.connections[rname])
        end
    end

    if mode == :nearest_neighbors
        k_int = Int(k)
        for rcell in receivers
            sorted = sort(sender_cells, by = s -> _cell_distance(rcell, s))
            for s in Iterators.take(sorted, k_int)
                connect!(model, rcell.name, s.name; release=release, w=w)
            end
        end

    elseif mode == :distance_inclusive
        k_dist = Float64(k)
        for rcell in receivers, s in sender_cells
            _cell_distance(rcell, s) <= k_dist &&
                connect!(model, rcell.name, s.name; release=release, w=w)
        end

    elseif mode == :distance_exclusive
        k_dist = Float64(k)
        for rcell in receivers
            in_range = filter(s -> _cell_distance(rcell, s) <= k_dist, sender_cells)
            isempty(in_range) && continue
            closest = in_range[argmin([_cell_distance(rcell, s) for s in in_range])]
            connect!(model, rcell.name, closest.name; release=release, w=w)
        end

    else
        error("Unknown mode: $mode. Choose :nearest_neighbors, :distance_inclusive, or :distance_exclusive")
    end

    return model
end
