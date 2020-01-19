__precompile__(false)
module PoreFlow 

    using SparseArrays
    using LinearAlgebra
    using PyCall
    using PyPlot
    using Parameters
    np = pyimport("numpy")
    if Sys.isapple()
        matplotlib.use("macosx")
    end

    pts = @. ([-1/sqrt(3); 1/sqrt(3)] + 1)/2

    include("Struct.jl")
    include("Core.jl")
    include("Visualization.jl")

end