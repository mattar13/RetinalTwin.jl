using CSV

"""
    load_stimulus_table(csv_path::AbstractString)

Load a stimulus table CSV (produced by `ElectroPhysiology.save_stimulus_csv`) and
return a `Vector{NamedTuple}` with fields `intensity` and `duration`, ready for
use with `simulate_erg(model, u0, params, stim_model; ...)`.

Expected CSV columns: `Type`, `Intensity`, `Duration` (and optionally
`CumulativeIntensity`, `TimeStart`, `TimeEnd`).
"""
function load_stimulus_table(csv_path::AbstractString)
    rows = CSV.File(csv_path)
    stim_model = NamedTuple{(:intensity, :duration), Tuple{Float64, Float64}}[]
    for row in rows
        intensity = Float64(getproperty(row, :Intensity))
        duration  = Float64(getproperty(row, :Duration))
        push!(stim_model, (intensity=intensity, duration=duration))
    end
    return stim_model
end
