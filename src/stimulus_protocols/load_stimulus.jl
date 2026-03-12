using CSV

"""
    load_stimulus_table(csv_path::AbstractString)

Load a stimulus table CSV (produced by `ElectroPhysiology.save_stimulus_csv`) and
return a `Vector{NamedTuple}` with fields `intensity`, `duration`, `time_start`,
and `time_end`.

Expected CSV columns: `Type`, `Intensity`, `Duration`, `TimeStart`, `TimeEnd`
(and optionally `CumulativeIntensity`).
"""
function load_stimulus_table(csv_path::AbstractString)
    rows = CSV.File(csv_path)
    stim_model = NamedTuple{(:intensity, :duration, :time_start, :time_end),
                            Tuple{Float64, Float64, Float64, Float64}}[]
    for row in rows
        intensity  = Float64(getproperty(row, :Intensity))
        duration   = Float64(getproperty(row, :Duration))
        time_start = hasproperty(row, :TimeStart) ? Float64(getproperty(row, :TimeStart)) : 0.0
        time_end   = hasproperty(row, :TimeEnd)   ? Float64(getproperty(row, :TimeEnd))   : duration
        push!(stim_model, (intensity=intensity, duration=duration,
                           time_start=time_start, time_end=time_end))
    end
    return stim_model
end
