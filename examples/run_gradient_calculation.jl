using Revise
using RetinalTwin

println("=" ^ 60)
println("RetinalTwin IR Gradient Calculation")
println("=" ^ 60)

map_path = joinpath(@__DIR__, "data", "column_map.json")
if isfile(map_path)
    model, u0 = load_mapping(map_path)
else
    pc_coords = square_grid_coords(16)
    model, u0 = build_column(; nPC=16, nHC=4, nONBC=4, nOFFBC=0, nA2=4, nGC=1, nMG=4, pc_coords=pc_coords)
    save_mapping(map_path, model, u0)
end

alpha = 1e-3
# target = nothing
target = "ON_BIPOLAR_PARAMS.g_CaL"
# target = ["ON_BIPOLAR_PARAMS.g_CaL", "ON_BIPOLAR_PARAMS.g_K", (Symbol("A2_AMACRINE_PARAMS"), Symbol("g_gap"))]
# outputs = :ONBC                 # one output type
outputs = [:ONBC, :A2, :GC]     # multiple output types
# outputs = [:ONBC1, :ONBC2]      # multiple specific cells
# outputs = ["ONBC_pool" => [:ONBC1, :ONBC2, :ONBC3, :ONBC4], :GC]  # grouped + standard output
# outputs = nothing                 # default: all cell types
out_csv = joinpath(@__DIR__, "data", "on_bipolar_g_CaL_gradient.csv")

rows = run_gradient_calculation(
    model, u0;
    target = target,
    alpha=alpha,
    outputs=outputs,
    out_csv=out_csv,
)

println("Saved gradient CSV: ", out_csv)
println("Rows: ", length(rows))
