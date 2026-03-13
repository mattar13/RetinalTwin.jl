#!/usr/bin/env julia
#
# Build a precompiled sysimage for RetinalTwin CLI scripts.
# This eliminates first-run compilation overhead when calling run_simulation.jl.
#
# Usage:
#   julia --project scripts/build_sysimage.jl
#
# Then run simulations with:
#   julia --project --sysimage retinal_twin.{so,dll,dylib} scripts/run_simulation.jl --outdir results/

using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

# Ensure PackageCompiler is available
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    @info "Adding PackageCompiler to project..."
    Pkg.add("PackageCompiler")
end

using PackageCompiler

# Platform-appropriate shared library extension
sysimage_ext = Sys.iswindows() ? "dll" : (Sys.isapple() ? "dylib" : "so")
sysimage_path = joinpath(dirname(@__DIR__), "retinal_twin.$sysimage_ext")
precompile_script = joinpath(@__DIR__, "run_simulation.jl")

packages = [
    :RetinalTwin,
    :DifferentialEquations,
    :CSV,
    :DataFrames,
    :ArgParse,
]

@info "Building sysimage..." sysimage_path packages
@info "This may take several minutes on the first run."

create_sysimage(
    packages;
    sysimage_path = sysimage_path,
    precompile_execution_file = precompile_script,
)

@info "Sysimage built successfully!" sysimage_path
println("\nRun simulations with:")
println("  julia --project --sysimage $sysimage_path scripts/run_simulation.jl --outdir results/")
