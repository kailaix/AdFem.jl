export PDATA, get_edge_dof

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
    Mesh(filename::String; file_format::Union{String, Missing} = missing)

Reads a mesh file and extracts element, coordinates and boundaries.

# Example
```julia
mesh = Mesh(joinpath(PDATA, "twoholes.stl"))
```
"""
function Mesh(filename::String; file_format::Union{String, Missing} = missing, 
                order::Int64 = 2, degree::Int64 = 1)
    meshio = get_meshio()
    if !ismissing(file_format)
        mesh = meshio.read(filename, file_format = file_format)
    else
        mesh = meshio.read(filename)
    end
    Mesh(Float64.(mesh.points[:,1:2]), Int64.(mesh.cells[1][2]) .+ 1, order, degree)
end

"""
    get_edge_dof(edges::Array{Int64, 2}, mesh::Mesh)
    get_edge_dof(edges::Array{Int64, 1}, mesh::Mesh)

Returns the DOFs for `edges`, which is a `K × 2` array containing vertex indices. 
"""
function get_edge_dof(edges::Array{Int64, 2}, mesh::Mesh)
    d = Dict{Tuple{Int64, Int64}, Int64}()
    for i = 1:mesh.nedge
        d[(mesh.edges[i, 1], mesh.edges[i, 2])] = i 
    end
    idx = Int64[]
    for i = 1:size(edges, 1)
        m = minimum(edges[i,:])
        M = maximum(edges[i,:])
        push!(idx, d[(m, M)])
    end
    idx
end

function get_edge_dof(edges::Array{Int64, 1}, mesh::Mesh)
    idx = get_edge_dof(reshape(edges, 1, 2), mesh)
    return idx[1]
end