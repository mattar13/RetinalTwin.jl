module DigitalTwin

#Export necessary packages for integrating physiological data
using ElectroPhysiology #This is for opening data
export getStimulusStartTime, getStimulusEndTime

#Export necessary packages for modeling
using DifferentialEquations
export solve, ODEProblem, Tsit5
export SteadyStateProblem

#Export basic optimization packages
using Optimization
export OprimizationProblem

#Export some more specialized optimization packages
using OptimizationBBO, OptimizationPRIMA, OptimizationOptimJL
export BBO_adaptive_de_rand_1_bin_radiuslimited, PRIMA, COBYLA, OptimizationOptimJL

using SciMLSensitivity #This is for sensitivity analysis

#Use some statistics packages
using Statistics

#When parameters have been found, save their results
using DataFrames, CSV

#Export some basic plotting packages
using GLMakie, PhysiologyPlotting

include("OpenData.jl")
export openData

include("AuxillaryFunctions.jl")
export Stim

include("Models.jl")
export make_model, make_model_photo
export simulate_model, simulate_model_photo

include("LossFunctions.jl")
export loss_static, loss_static_abm, loss_graded


end