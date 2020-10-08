module PoreFlow 

    using SparseArrays
    using LinearAlgebra
    using PyCall
    using PyPlot
    using Parameters
    using Reexport
    using Statistics
    using MAT
    @reexport using ADCME

    pts = @. ([-1/sqrt(3); 1/sqrt(3)] + 1)/2
    np = PyNULL()
    interpolate = PyNULL()
    LIBMFEM = abspath(joinpath(@__DIR__, "..",  "deps", "MFEM", "build", get_library_name("nnfem_mfem")))
    libmfem = missing 

    function __init__()
        copy!(np, pyimport("numpy"))
        copy!(interpolate,pyimport("scipy.interpolate"))
        global libmfem = tf.load_op_library(LIBMFEM) # load for tensorflow first
    end

    include("Struct.jl")
    include("Utils.jl")
    include("Core.jl")
    include("Plasticity.jl")
    include("InvCore.jl")
    include("Viscoelasticity.jl")
    include("Visualization.jl")
    include("Constitutive.jl")
    include("Solver.jl")
    include("MFEM/MFEM.jl")
    include("MFEM/MCore.jl")
    include("MFEM/MVisualize.jl")
    include("MFEM/MUtils.jl")
    include("MFEM/Mechanics.jl")
    include("MFEM/MBDM.jl")
    

end