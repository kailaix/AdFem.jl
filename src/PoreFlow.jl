module PoreFlow 

    using SparseArrays
    using LinearAlgebra
    using PyCall
    using PyPlot
    using Parameters
    using Reexport
    using Statistics
    @reexport using ADCME

    pts = @. ([-1/sqrt(3); 1/sqrt(3)] + 1)/2
    np = PyNULL()
    interpolate = PyNULL()
    function __init__()
        global np, interpolate
        np = pyimport("numpy")
        interpolate = pyimport("scipy.interpolate")
    end

    include("Struct.jl")
    include("Utils.jl")
    include("Core.jl")
    include("Plasticity.jl")
    include("InvCore.jl")
    include("Viscoelasticity.jl")
    include("Visualization.jl")
    

end