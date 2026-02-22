# ============================================================
# parameter_extraction.jl - Extract parameters from CSV files
# ============================================================

using CSV
using DataFrames

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

Load photoreceptor parameters from a CSV file and return as a NamedTuple.
The CSV should have columns: Key, Value, LowerBounds, UpperBounds, DEFAULT

# Arguments
- `csv_path`: Path to the CSV file
- `editable`: If `true`, return a mutable `Dict{Symbol,Any}` instead of a `NamedTuple`

# Returns
- Parameter collection keyed by symbol names (`NamedTuple` by default, `Dict` when editable)
"""
function load_params_from_csv(csv_path::String; editable::Bool=false)
    # Read CSV file
    df = CSV.read(csv_path, DataFrame)

    # Extract parameter names (as symbols) and values
    param_names = Symbol.(df.Key)
    param_values = df.Value

    params_nt = NamedTuple{Tuple(param_names)}(param_values)
    return editable ? namedtuple_to_dict(params_nt) : params_nt
end

"""
    default_rod_params()

Load default rod photoreceptor parameters from the bundled CSV file.
"""
function default_rod_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "photoreceptor_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_hc_params()

Load default horizontal cell parameters from the bundled CSV file.
"""
function default_hc_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "horizontal_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_on_bc_params()

Load default ON bipolar cell parameters from the bundled CSV file.
"""
function default_on_bc_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "on_bipolar_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_off_bc_params()

Load default OFF bipolar cell parameters from the bundled CSV file.
"""
function default_off_bc_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "off_bipolar_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_a2_params()

Load default A2 amacrine cell parameters from the bundled CSV file.
"""
function default_a2_amacrine_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "a2_amacrine_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_gaba_params()

Load default GABAergic amacrine cell parameters from the bundled CSV file.
"""
function default_gaba_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "gaba_amacrine_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_da_params()

Load default dopaminergic amacrine cell parameters from the bundled CSV file.
"""
function default_da_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "da_amacrine_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_gc_params()

Load default ganglion cell parameters from the bundled CSV file.
"""
function default_gc_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "ganglion_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_muller_params()

Load default Müller glial cell parameters from the bundled CSV file.
"""
function default_muller_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "muller_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

"""
    default_rpe_params()

Load default RPE cell parameters from the bundled CSV file.
"""
function default_rpe_params(; editable::Bool=false)
    csv_path = joinpath(@__DIR__, "rpe_params.csv")
    return load_params_from_csv(csv_path; editable=editable)
end

default_a2_params(; editable::Bool=false) = default_a2_amacrine_params(; editable=editable)

# ============================================================
# Global parameter and state management
# ============================================================

"""
    load_all_params()

Load all cell type parameters from CSV files and return as a single NamedTuple.

# Returns
A NamedTuple with fields:
- `rod`: Rod photoreceptor parameters
- `hc`: Horizontal cell parameters
- `on_bc`: ON bipolar cell parameters
- `off_bc`: OFF bipolar cell parameters
- `a2`: A2 amacrine cell parameters
- `gaba`: GABAergic amacrine cell parameters
- `da`: Dopaminergic amacrine cell parameters
- `gc`: Ganglion cell parameters
- `muller`: Müller glial cell parameters
- `rpe`: RPE cell parameters

# Example
```julia
params = load_all_params()
rod_C_m = params.rod.C_m
on_bc_g_Ca = params.on_bc.g_Ca
```
"""
function load_all_params(; editable::Bool=false)
    params_nt = (
        rod = default_rod_params(; editable=false),
        hc = default_hc_params(; editable=false),
        on_bc = default_on_bc_params(; editable=false),
        off_bc = default_off_bc_params(; editable=false),
        a2 = default_a2_params(; editable=false),
        gaba = default_gaba_params(; editable=false),
        da = default_da_params(; editable=false),
        gc = default_gc_params(; editable=false),
        muller = default_muller_params(; editable=false),
        rpe = default_rpe_params(; editable=false)
    )
    return editable ? namedtuple_to_dict(params_nt) : params_nt
end
