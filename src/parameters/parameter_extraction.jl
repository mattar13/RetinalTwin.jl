# ============================================================
# parameter_extraction.jl - Extract parameters from CSV files
# ============================================================

using CSV
using DataFrames

"""
    load_photoreceptor_params_from_csv(csv_path::String)

Load photoreceptor parameters from a CSV file and return as a NamedTuple.
The CSV should have columns: Key, Value, LowerBounds, UpperBounds, DEFAULT

# Arguments
- `csv_path`: Path to the CSV file

# Returns
- Named tuple with parameter names as symbols and values as numbers
"""
function load_params_from_csv(csv_path::String)
    # Read CSV file
    df = CSV.read(csv_path, DataFrame)

    # Extract parameter names (as symbols) and values
    param_names = Symbol.(df.Key)
    param_values = df.Value

    # Create NamedTuple
    return NamedTuple{Tuple(param_names)}(param_values)
end

"""
    default_rod_params()

Load default rod photoreceptor parameters from the bundled CSV file.
"""
function default_rod_params()
    csv_path = joinpath(@__DIR__, "photoreceptor_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_hc_params()

Load default horizontal cell parameters from the bundled CSV file.
"""
function default_hc_params()
    csv_path = joinpath(@__DIR__, "horizontal_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_on_bc_params()

Load default ON bipolar cell parameters from the bundled CSV file.
"""
function default_on_bc_params()
    csv_path = joinpath(@__DIR__, "on_bipolar_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_off_bc_params()

Load default OFF bipolar cell parameters from the bundled CSV file.
"""
function default_off_bc_params()
    csv_path = joinpath(@__DIR__, "off_bipolar_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_a2_params()

Load default A2 amacrine cell parameters from the bundled CSV file.
"""
function default_a2_amacrine_params()
    csv_path = joinpath(@__DIR__, "a2_amacrine_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_gaba_params()

Load default GABAergic amacrine cell parameters from the bundled CSV file.
"""
function default_gaba_params()
    csv_path = joinpath(@__DIR__, "gaba_amacrine_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_da_params()

Load default dopaminergic amacrine cell parameters from the bundled CSV file.
"""
function default_da_params()
    csv_path = joinpath(@__DIR__, "da_amacrine_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_gc_params()

Load default ganglion cell parameters from the bundled CSV file.
"""
function default_gc_params()
    csv_path = joinpath(@__DIR__, "ganglion_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_muller_params()

Load default Müller glial cell parameters from the bundled CSV file.
"""
function default_muller_params()
    csv_path = joinpath(@__DIR__, "muller_params.csv")
    return load_params_from_csv(csv_path)
end

"""
    default_rpe_params()

Load default RPE cell parameters from the bundled CSV file.
"""
function default_rpe_params()
    csv_path = joinpath(@__DIR__, "rpe_params.csv")
    return load_params_from_csv(csv_path)
end

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
function load_all_params()
    return (
        rod = default_rod_params(),
        hc = default_hc_params(),
        on_bc = default_on_bc_params(),
        off_bc = default_off_bc_params(),
        a2 = default_a2_params(),
        gaba = default_gaba_params(),
        da = default_da_params(),
        gc = default_gc_params(),
        muller = default_muller_params(),
        rpe = default_rpe_params()
    )
end
