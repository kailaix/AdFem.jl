export PDATA 

"""
    PDATA

The folder where built-in meshes are stored. 
"""
PDATA = joinpath(@__DIR__, "..", "..",  "deps", "MFEM", "MeshData")


function get_meshio()
    try
        pyimport("meshio")
    catch
        PIP = joinpath(ADCME.BINDIR, "pip")
        run_with_env(`$PIP install meshio`)
        pyimport("meshio")
    end
end

"""
    Mesh(stlfile::String)

Reads a mesh file and extracts element, coordinates and boundaries.

# Example
```julia
mesh = Mesh(joinpath(PDATA, "twoholes.stl"))
```
"""
function Mesh(filename::String)
    meshio = get_meshio()
    mesh = meshio.read(filename)
    Mesh(mesh.points[:,1:2], mesh.cells[1][2] .+ 1)
end