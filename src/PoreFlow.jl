__precompile__(false)
module PoreFlow 

    using SparseArrays
    using PyPlot
    using LinearAlgebra
    using PyCall
    np = pyimport("numpy")
    matplotlib.use("macosx")

    pts = @. ([-1/sqrt(3); 1/sqrt(3)] + 1)/2

    include("Core.jl")

end