__precompile__(false)
module PoreFlow 

    using SparseArrays
    using LinearAlgebra
    using PyCall
    using PyPlot
    using Parameters
    np = pyimport("numpy")
    interpolate = pyimport("scipy.interpolate")

    pts = @. ([-1/sqrt(3); 1/sqrt(3)] + 1)/2

    include("Struct.jl")
    include("Utils.jl")
    include("Core.jl")
    include("Plasticity.jl")
    include("Visualization.jl")

end