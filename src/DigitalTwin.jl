module DigitalTwin

using ElectroPhysiology #This is for opening data
using DifferentialEquations

using Optimization
export OprimizationProblem

using OptimizationBBO, OptimizationPRIMA, OptimizationOptimJL
export BBO_adaptive_de_rand_1_bin_radiuslimited, PRIMA, COBYLA, OptimizationOptimJL

using DiffEqParamEstim #This is for parameter estimation

using Statistics


using DataFrames, CSV
using GLMakie, PhysiologyPlotting

include("OpenData.jl")

include("AuxillaryFunctions.jl")
export Stim

include("Models.jl")
export make_model, make_model_photo
export simulate_model, simulate_model_photo

include("LossFunctions.jl")
export loss_static, loss_static_abm, loss_graded


end