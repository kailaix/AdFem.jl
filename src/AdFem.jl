module AdFem 

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
    pv = PyNULL()
    LIBMFEM = abspath(joinpath(@__DIR__, "..",  "deps", "MFEM", "build", get_library_name("admfem")))
    LIBMFEM3 = abspath(joinpath(@__DIR__, "..",  "deps", "MFEM3", "build", get_library_name("admfem")))
    libmfem = missing 
    libmfem3 = missing 
    LIBADFEM = abspath(joinpath(@__DIR__, "..",  "deps", "build", get_library_name("adfem")))
    libadfem = missing

    function __init__()
        copy!(np, pyimport("numpy"))
        try 
            copy!(pv, pyimport("pyvista"))
        catch
            @warn """pyvista installation was not successful. 3D plots functionalities are disabled.
To fix the problem, check why `$(ADCME.get_pip()) install pyvista` failed."""
        end
        if !isfile(LIBMFEM) || !isfile(LIBADFEM) || !isfile(LIBMFEM3)
            error("Dependencies of AdFem not properly built. Run `Pkg.build(\"AdFem\")` to rebuild AdFem.")
        end
        global libmfem = load_library(LIBMFEM)
        global libmfem3 = load_library(LIBMFEM3)
        global libadfem = load_library(LIBADFEM)
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
    include("MFEM3/MFEM.jl")
    include("MFEM3/MCore.jl")
    include("MFEM3/MVisualize.jl")
    include("pcl.jl")

end