# ============================================================
# parameter_extraction.jl - Extract parameters from CSV files
# ============================================================

using CSV
using DataFrames

struct ParameterSpec
    value::Float64
    lower::Float64
    upper::Float64
    fixed::Bool
    description::String
end

default_param_csv_path() = joinpath(@__DIR__, "retinal_params.csv")

const _CELL_TYPE_ALIASES = Dict{Symbol,Symbol}(
    :ROD => :PHOTORECEPTOR_PARAMS,
    :PHOTORECEPTOR => :PHOTORECEPTOR_PARAMS,
    :PHOTORECEPTOR_PARAMS => :PHOTORECEPTOR_PARAMS,
    :HORIZONTAL => :HORIZONTAL_PARAMS,
    :HORIZONTAL_PARAMS => :HORIZONTAL_PARAMS,
    :ON_BIPOLAR => :ON_BIPOLAR_PARAMS,
    :ON_BIPOLAR_PARAMS => :ON_BIPOLAR_PARAMS,
    :OFF_BIPOLAR => :OFF_BIPOLAR_PARAMS,
    :OFF_BIPOLAR_PARAMS => :OFF_BIPOLAR_PARAMS,
    :A2 => :A2_AMACRINE_PARAMS,
    :A2_AMACRINE => :A2_AMACRINE_PARAMS,
    :A2_AMACRINE_PARAMS => :A2_AMACRINE_PARAMS,
    :GABA => :GABA_AMACRINE_PARAMS,
    :GABA_AMACRINE => :GABA_AMACRINE_PARAMS,
    :GABA_AMACRINE_PARAMS => :GABA_AMACRINE_PARAMS,
    :DA => :DA_AMACRINE_PARAMS,
    :DA_AMACRINE => :DA_AMACRINE_PARAMS,
    :DA_AMACRINE_PARAMS => :DA_AMACRINE_PARAMS,
    :GANGLION => :GANGLION_PARAMS,
    :GANGLION_PARAMS => :GANGLION_PARAMS,
    :MULLER => :MULLER_PARAMS,
    :MULLER_PARAMS => :MULLER_PARAMS,
    :RPE => :RPE_PARAMS,
    :RPE_PARAMS => :RPE_PARAMS,
)

_normalize_cell_type(s::Symbol) = get(_CELL_TYPE_ALIASES, Symbol(uppercase(String(s))), Symbol(uppercase(String(s))))

function _parse_float_or(x, fallback::Float64)
    x === missing && return fallback
    s = strip(String(x))
    isempty(s) && return fallback
    y = tryparse(Float64, s)
    return y === nothing ? fallback : y
end

function _parse_bool(x, fallback::Bool=false)
    x === missing && return fallback
    s = lowercase(strip(String(x)))
    isempty(s) && return fallback
    s in ("true", "t", "1", "yes", "y") && return true
    s in ("false", "f", "0", "no", "n") && return false
    return fallback
end

function _ten_percent_bounds(v::Float64)
    lo = 0.9 * v
    hi = 1.1 * v
    return min(lo, hi), max(lo, hi)
end

function _rows_for_cell_type(df::DataFrame, cell_type::Union{Nothing,Symbol})
    cell_type === nothing && return eachrow(df)
    if :CellType in propertynames(df)
        wanted = _normalize_cell_type(cell_type)
        return (row for row in eachrow(df) if _normalize_cell_type(Symbol(uppercase(strip(String(getproperty(row, :CellType)))))) == wanted)
    end
    return eachrow(df)
end

function _specs_from_dataframe(df::DataFrame; cell_type::Union{Nothing,Symbol}=nothing, editable::Bool=false)
    names = Symbol[]
    specs = ParameterSpec[]
    rows = _rows_for_cell_type(df, cell_type)
    for row in rows
        key = Symbol(strip(String(getproperty(row, :Key))))
        value = _parse_float_or(getproperty(row, :Value), NaN)
        isnan(value) && continue

        default_lo, default_hi = _ten_percent_bounds(value)
        lower =
            :LowerBounds in propertynames(df) ? _parse_float_or(getproperty(row, :LowerBounds), default_lo) :
            (:LowerBound in propertynames(df) ? _parse_float_or(getproperty(row, :LowerBound), default_lo) : default_lo)
        upper =
            :UpperBounds in propertynames(df) ? _parse_float_or(getproperty(row, :UpperBounds), default_hi) :
            (:UpperBound in propertynames(df) ? _parse_float_or(getproperty(row, :UpperBound), default_hi) : default_hi)
        lo = min(lower, upper)
        hi = max(lower, upper)
        fixed = :Fixed in propertynames(df) ? _parse_bool(getproperty(row, :Fixed), false) : false
        desc = :Description in propertynames(df) ? strip(String(getproperty(row, :Description))) : ""

        push!(names, key)
        push!(specs, ParameterSpec(value, lo, hi, fixed, desc))
    end

    specs_nt = NamedTuple{Tuple(names)}(Tuple(specs))
    return editable ? namedtuple_to_dict(specs_nt; recursive=false) : specs_nt
end

function _values_from_specs(specs::NamedTuple; editable::Bool=false)
    names = collect(keys(specs))
    values = [getproperty(specs, k).value for k in names]
    params_nt = NamedTuple{Tuple(names)}(Tuple(values))
    return editable ? namedtuple_to_dict(params_nt; recursive=false) : params_nt
end

function _all_specs_from_dataframe(df::DataFrame; editable::Bool=false)
    cell_order = Symbol[]
    grouped_names = Dict{Symbol,Vector{Symbol}}()
    grouped_specs = Dict{Symbol,Vector{ParameterSpec}}()
    has_cell_type = :CellType in propertynames(df)

    for row in eachrow(df)
        key = Symbol(strip(String(getproperty(row, :Key))))
        value = _parse_float_or(getproperty(row, :Value), NaN)
        isnan(value) && continue

        default_lo, default_hi = _ten_percent_bounds(value)
        lower =
            :LowerBounds in propertynames(df) ? _parse_float_or(getproperty(row, :LowerBounds), default_lo) :
            (:LowerBound in propertynames(df) ? _parse_float_or(getproperty(row, :LowerBound), default_lo) : default_lo)
        upper =
            :UpperBounds in propertynames(df) ? _parse_float_or(getproperty(row, :UpperBounds), default_hi) :
            (:UpperBound in propertynames(df) ? _parse_float_or(getproperty(row, :UpperBound), default_hi) : default_hi)
        lo = min(lower, upper)
        hi = max(lower, upper)
        fixed = :Fixed in propertynames(df) ? _parse_bool(getproperty(row, :Fixed), false) : false
        desc = :Description in propertynames(df) ? strip(String(getproperty(row, :Description))) : ""

        cell_type = has_cell_type ? _normalize_cell_type(Symbol(uppercase(strip(String(getproperty(row, :CellType)))))) : :PARAMS
        if !(cell_type in cell_order)
            push!(cell_order, cell_type)
            grouped_names[cell_type] = Symbol[]
            grouped_specs[cell_type] = ParameterSpec[]
        end
        push!(grouped_names[cell_type], key)
        push!(grouped_specs[cell_type], ParameterSpec(value, lo, hi, fixed, desc))
    end

    out_names = Symbol[]
    out_values = NamedTuple[]
    for ct in cell_order
        inner = NamedTuple{Tuple(grouped_names[ct])}(Tuple(grouped_specs[ct]))
        push!(out_names, ct)
        push!(out_values, inner)
    end
    specs_nt = NamedTuple{Tuple(out_names)}(Tuple(out_values))
    return editable ? namedtuple_to_dict(specs_nt) : specs_nt
end

"""
    dict_to_namedtuple(d::AbstractDict; recursive::Bool=true)

Convert a dictionary to a `NamedTuple`. By default, nested dictionaries are
converted recursively.
"""
function dict_to_namedtuple(d::AbstractDict; recursive::Bool=true)
    entries = collect(pairs(d))
    keys_tuple = Tuple(Symbol(k) for (k, _) in entries)
    values_tuple = Tuple(
        recursive && (v isa AbstractDict) ? dict_to_namedtuple(v; recursive=true) : v
        for (_, v) in entries
    )
    return NamedTuple{keys_tuple}(values_tuple)
end

"""
    namedtuple_to_dict(nt::NamedTuple; recursive::Bool=true)

Convert a `NamedTuple` to a mutable `Dict{Symbol,Any}`. By default, nested
`NamedTuple`s are converted recursively.
"""
function namedtuple_to_dict(nt::NamedTuple; recursive::Bool=true)
    out = Dict{Symbol,Any}()
    for k in keys(nt)
        v = getproperty(nt, k)
        out[k] = recursive && (v isa NamedTuple) ? namedtuple_to_dict(v; recursive=true) : v
    end
    return out
end

"""
    load_params_from_csv(csv_path::String; editable::Bool=false)

Load parameter values from a CSV file and return them as a NamedTuple.
For unified files with multiple cell types, pass `cell_type=...`.

# Arguments
- `csv_path`: Path to the CSV file
- `editable`: If `true`, return a mutable `Dict{Symbol,Any}` instead of a `NamedTuple`
- `cell_type`: Optional cell-type selector for unified CSV layouts

# Returns
- Parameter collection keyed by symbol names (`NamedTuple` by default, `Dict` when editable)
"""
function load_params_from_csv(csv_path::String; editable::Bool=false, cell_type::Union{Nothing,Symbol}=nothing)
    df = CSV.read(csv_path, DataFrame)
    if cell_type === nothing && (:CellType in propertynames(df))
        types_present = unique(_normalize_cell_type(Symbol(uppercase(strip(String(v))))) for v in df.CellType if !ismissing(v))
        length(types_present) > 1 && error("CSV contains multiple cell types. Pass `cell_type=...` when loading.")
    end
    specs = _specs_from_dataframe(df; cell_type=cell_type, editable=false)
    return _values_from_specs(specs; editable=editable)
end

function load_param_specs_from_csv(csv_path::String; editable::Bool=false, cell_type::Union{Nothing,Symbol}=nothing)
    df = CSV.read(csv_path, DataFrame)
    if cell_type === nothing && (:CellType in propertynames(df))
        types_present = unique(_normalize_cell_type(Symbol(uppercase(strip(String(v))))) for v in df.CellType if !ismissing(v))
        length(types_present) > 1 && error("CSV contains multiple cell types. Pass `cell_type=...` when loading.")
    end
    return _specs_from_dataframe(df; cell_type=cell_type, editable=editable)
end

function load_all_param_specs_from_csv(csv_path::String; editable::Bool=false)
    df = CSV.read(csv_path, DataFrame)
    return _all_specs_from_dataframe(df; editable=editable)
end

function load_all_param_specs(; editable::Bool=false)
    return load_all_param_specs_from_csv(default_param_csv_path(); editable=editable)
end

function get_param_spec(cell_type::Symbol, key::Symbol; csv_path::String=default_param_csv_path())
    specs_by_cell = load_all_param_specs_from_csv(csv_path; editable=false)
    ct = _normalize_cell_type(cell_type)
    hasproperty(specs_by_cell, ct) || return nothing
    specs = getproperty(specs_by_cell, ct)
    hasproperty(specs, key) || return nothing
    return getproperty(specs, key)
end

"""
    default_rod_params()

Load default rod photoreceptor parameters from the bundled CSV file.
"""
function default_rod_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    rod_params = all_params.PHOTORECEPTOR_PARAMS
    return editable ? namedtuple_to_dict(rod_params; recursive=false) : rod_params
end

"""
    default_hc_params()

Load default horizontal cell parameters from the bundled CSV file.
"""
function default_hc_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    hc_params = all_params.HORIZONTAL_PARAMS
    return editable ? namedtuple_to_dict(hc_params; recursive=false) : hc_params
end

"""
    default_on_bc_params()

Load default ON bipolar cell parameters from the bundled CSV file.
"""
function default_on_bc_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    on_params = all_params.ON_BIPOLAR_PARAMS
    return editable ? namedtuple_to_dict(on_params; recursive=false) : on_params
end

"""
    default_off_bc_params()

Load default OFF bipolar cell parameters from the bundled CSV file.
"""
function default_off_bc_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    off_params = all_params.OFF_BIPOLAR_PARAMS
    return editable ? namedtuple_to_dict(off_params; recursive=false) : off_params
end

"""
    default_a2_params()

Load default A2 amacrine cell parameters from the bundled CSV file.
"""
function default_a2_amacrine_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    a2_params = all_params.A2_AMACRINE_PARAMS
    return editable ? namedtuple_to_dict(a2_params; recursive=false) : a2_params
end

"""
    default_gaba_params()

Load default GABAergic amacrine cell parameters from the bundled CSV file.
"""
function default_gaba_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    gaba_params = all_params.GABA_AMACRINE_PARAMS
    return editable ? namedtuple_to_dict(gaba_params; recursive=false) : gaba_params
end

"""
    default_da_params()

Load default dopaminergic amacrine cell parameters from the bundled CSV file.
"""
function default_da_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    da_params = all_params.DA_AMACRINE_PARAMS
    return editable ? namedtuple_to_dict(da_params; recursive=false) : da_params
end

"""
    default_gc_params()

Load default ganglion cell parameters from the bundled CSV file.
"""
function default_gc_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    gc_params = all_params.GANGLION_PARAMS
    return editable ? namedtuple_to_dict(gc_params; recursive=false) : gc_params
end

"""
    default_muller_params()

Load default Müller glial cell parameters from the bundled CSV file.
"""
function default_muller_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    muller_params = all_params.MULLER_PARAMS
    return editable ? namedtuple_to_dict(muller_params; recursive=false) : muller_params
end

"""
    default_rpe_params()

Load default RPE cell parameters from the bundled CSV file.
"""
function default_rpe_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    rpe_params = all_params.RPE_PARAMS
    return editable ? namedtuple_to_dict(rpe_params; recursive=false) : rpe_params
end

default_a2_params(; editable::Bool=false) = default_a2_amacrine_params(; editable=editable)

# ============================================================
# Global parameter and state management
# ============================================================

"""
    load_all_params()

Load all cell type parameters from the unified CSV and return as a single
`NamedTuple` of `NamedTuple`s.

# Returns
A NamedTuple keyed by normalized `CellType` symbols (for example,
`PHOTORECEPTOR_PARAMS`, `ON_BIPOLAR_PARAMS`, `A2_AMACRINE_PARAMS`, etc).
Each entry is a parameter-value NamedTuple for that cell type.

# Example
```julia
params = load_all_params()
rod_C_m = params.PHOTORECEPTOR_PARAMS.C_m
on_bc_g_Ca = params.ON_BIPOLAR_PARAMS.g_CaL
```
"""
function load_all_params(; editable::Bool=false)
    specs_by_cell = load_all_param_specs(; editable=false)
    cell_names = collect(keys(specs_by_cell))
    values = [_values_from_specs(getproperty(specs_by_cell, ct); editable=false) for ct in cell_names]
    params_nt = NamedTuple{Tuple(cell_names)}(Tuple(values))
    return editable ? namedtuple_to_dict(params_nt) : params_nt
end
