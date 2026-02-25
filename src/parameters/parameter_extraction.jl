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
    :PHOTO => :PHOTO,
    :HC => :HC,
    :ONBC => :ONBC,
    :OFFBC => :OFFBC,
    :A2 => :A2,
    :GABA => :GABA,
    :DA => :DA,
    :GC => :GC,
    :MULLER => :MULLER,
    :RPE => :RPE,
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

function load_all_param_specs(; editable::Bool=false, csv_path::String=default_param_csv_path())
    df = CSV.read(csv_path, DataFrame)
    specs_by_cell = _all_specs_from_dataframe(df; editable=false)
    return editable ? namedtuple_to_dict(specs_by_cell) : specs_by_cell
end

function get_param_spec(cell_type::Symbol, key::Symbol; csv_path::String=default_param_csv_path())
    specs_by_cell = load_all_param_specs(csv_path=csv_path)
    ct = _normalize_cell_type(cell_type)
    hasproperty(specs_by_cell, ct) || return nothing
    specs = getproperty(specs_by_cell, ct)
    hasproperty(specs, key) || return nothing
    return getproperty(specs, key)
end

function load_all_params(; editable::Bool=false, csv_path::String=default_param_csv_path())
    specs_by_cell = load_all_param_specs(csv_path=csv_path)
    cell_names = collect(keys(specs_by_cell))
    values = [_values_from_specs(getproperty(specs_by_cell, ct); editable=false) for ct in cell_names]
    params_nt = NamedTuple{Tuple(cell_names)}(Tuple(values))
    return editable ? namedtuple_to_dict(params_nt) : params_nt
end
