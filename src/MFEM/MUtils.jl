export PDATA, get_edge_dof, impose_Dirichlet_boundary_conditions

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

@doc raw"""
    impose_Dirichlet_boundary_conditions(A::Union{SparseArrays, Array{Float64, 2}}, rhs::Array{Float64,1}, bdnode::Array{Int64, 1}, 
        bdval::Array{Float64,1})
    impose_Dirichlet_boundary_conditions(A::SparseTensor, rhs::Union{Array{Float64,1}, PyObject}, bdnode::Array{Int64, 1}, 
        bdval::Union{Array{Float64,1}, PyObject})

Algebraically impose the Dirichlet boundary conditions. We want the solutions at indices `bdnode` to be `bdval`. Given the matrix and the right hand side

$$\begin{bmatrix} A_{II} & A_{IB} \\ A_{BI} & A_{BB} \end{bmatrix}, \begin{bmatrix}r_I \\ r_B \end{bmatrix}$$

The function returns

$$\begin{bmatrix} A_{II} & 0 \\ 0 & I \end{bmatrix}, \begin{bmatrix}r_I - A_{IB} u_B \\ r_B \end{bmatrix}$$
"""
function impose_Dirichlet_boundary_conditions(A::Union{SparseMatrixCSC, Array{Float64, 2}}, rhs::Array{Float64,1}, bdnode::Array{Int64, 1}, 
    bdval::Array{Float64,1})
    N = length(rhs)
    r = copy(rhs)
    idx = ones(Bool, N)
    idx[bdnode] .= false
    A11 = A[idx, idx]
    A12 = A[idx, bdnode]
    r[idx] = r[idx] - A12 * bdval
    r[bdnode] = bdval 
    B = spzeros(N, N)
    B[idx, idx] = A11 
    B[bdnode, bdnode] = spdiagm(0=>ones(length(bdnode)))
    B, r
end

function impose_Dirichlet_boundary_conditions(A::SparseTensor, rhs::Union{Array{Float64,1}, PyObject}, bdnode::Array{Int64, 1}, 
    bdval::Union{Array{Float64,1}, PyObject})
    indices = A.o.indices 
    vv = A.o.values 
    impose_dirichlet_ = load_op_and_grad(PoreFlow.libmfem,"impose_dirichlet", multiple=true)
    indices,vv,bd,rhs,bdval = convert_to_tensor(Any[indices,vv,bdnode,rhs,bdval], [Int64,Float64,Int64,Float64,Float64])
    indices, vv, rhs = impose_dirichlet_(indices,vv,bd,rhs,bdval)
    RawSparseTensor(indices, vv, size(A)...), set_shape(rhs, (size(A,2),))
end