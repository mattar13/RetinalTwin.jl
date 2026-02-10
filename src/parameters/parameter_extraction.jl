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
function load_photoreceptor_params_from_csv(csv_path::String)
    # Read CSV file
    df = CSV.read(csv_path, DataFrame)

    # Extract parameter names (as symbols) and values
    param_names = Symbol.(df.Key)
    param_values = df.Value

    # Create NamedTuple
    return NamedTuple{Tuple(param_names)}(param_values)
end

"""
    default_rod_params_csv()

Load default rod photoreceptor parameters from the bundled CSV file.
"""
function default_rod_params_csv()
    # Construct path to the CSV file relative to this source file
    csv_path = joinpath(@__DIR__, "photoreceptor_params.csv")
    return load_photoreceptor_params_from_csv(csv_path)
end
