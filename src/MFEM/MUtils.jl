export PDATA, get_edge_dof, 
    impose_Dirichlet_boundary_conditions, dof_to_gauss_points, get_boundary_edge_orientation,
    compute_pml_term, solve_slip_law

"""
    PDATA

The folder where built-in meshes are stored. 
"""
PDATA = abspath(joinpath(@__DIR__, "..", "..",  "deps", "MFEM", "MeshData"))


function get_meshio()
    try
        pyimport("meshio")
    catch
        PIP = joinpath(ADCME.BINDIR, "pip")
        run_with_env(`$PIP install meshio==4.2`)
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
                order::Int64 = 2, degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = 2)
    if splitext(filename)[2] == ".mat"
        d = matread(filename)
        return Mesh(Float64.(d["nodes"]), Int64.(d["elems"]), order, degree, lorder)
    end
    meshio = AdFem.get_meshio()
    if !ismissing(file_format)
        mesh = meshio.read(filename, file_format = file_format)
    else
        mesh = meshio.read(filename)
    end
    elem = py"list($mesh.cells[0])"[2]
    if length(elem)==0
        error("No triangles found in the mesh file.")
    end
    Mesh(Float64.(mesh.points[:,1:2]), Int64.(elem) .+ 1, order, degree, lorder)
end

"""
    get_edge_normal(mmesh::Mesh)
"""
function get_edge_normal(mmesh::Mesh)
    _edge_to_elem_map = Set{Tuple{Int64, Int64}}()
    dic = (x,y)->(minimum([x,y]), maximum([x,y]))
    add_dic = (e1, e2)->begin 
        if dic(e1, e2) in _edge_to_elem_map
            delete!(_edge_to_elem_map, dic(e1, e2))
        else
            push!(_edge_to_elem_map, dic(e1, e2))
        end
    end
    edict = Dict()
    for i = 1:mmesh.nelem
        e1, e2, e3 = mmesh.elems[i,:]
        add_dic(e1, e2); edict[dic(e1,e2)] = e3
        add_dic(e3, e2); edict[dic(e3,e2)] = e1
        add_dic(e1, e3); edict[dic(e1,e3)] = e2
    end
    out = zeros(Float64, length(_edge_to_elem_map), 2)
    for (k,s) in enumerate(_edge_to_elem_map)
        p1 = s[1] 
        p2 = s[2]
        p3 = edict[s]
        x1 = mmesh.nodes[p1,:] - mmesh.nodes[p3,:]
        x2 = mmesh.nodes[p1,:] - mmesh.nodes[p2,:]
        n = [x2[2]; -x2[1]]
        n = n/norm(n)
        if dot(n, x1)<0
            n = -n 
        end
        out[k,:] = n
    end
    out
end

"""
    get_edge_normal(bdedge::Array{Int64, 2}, mmesh::Mesh)

Returns the outer normal for all edges in `bdedge`. 
"""
function get_edge_normal(bdedge::Array{Int64, 2}, mmesh::Mesh)
    _edge_to_elem_map = Set{Tuple{Int64, Int64}}()
    dic = (x,y)->(minimum([x,y]), maximum([x,y]))
    add_dic = (e1, e2)->begin 
        if dic(e1, e2) in _edge_to_elem_map
            delete!(_edge_to_elem_map, dic(e1, e2))
        else
            push!(_edge_to_elem_map, dic(e1, e2))
        end
    end
    edict = Dict()
    for i = 1:mmesh.nelem
        e1, e2, e3 = mmesh.elems[i,:]
        add_dic(e1, e2); edict[dic(e1,e2)] = e3
        add_dic(e3, e2); edict[dic(e3,e2)] = e1
        add_dic(e1, e3); edict[dic(e1,e3)] = e2
    end
    out = zeros(Float64, size(bdedge, 1), 2)
    for k = 1:size(bdedge, 1)
        s = bdedge[k,:]
        p1 = s[1] 
        p2 = s[2]
        p3 = edict[dic(p1, p2)]
        x1 = mmesh.nodes[p1,:] - mmesh.nodes[p3,:]
        x2 = mmesh.nodes[p1,:] - mmesh.nodes[p2,:]
        n = [x2[2]; -x2[1]]
        n = n/norm(n)
        if dot(n, x1)<0
            n = -n 
        end
        out[k,:] = n
    end
    out
end

"""
    save(filename::String, mesh::Mesh)

Saves the mesh to the file `filename`.
"""
function save(filename::String, mesh::Mesh)
    matwrite(filename, Dict(
        "nodes"=>mesh.nodes, "elems"=>mesh.elems
    ))
end

"""
    get_edge_dof(edges::Array{Int64, 2}, mesh::Mesh)
    get_edge_dof(edges::Array{Int64, 1}, mesh::Mesh)

Returns the DOFs for `edges`, which is a `K × 2` array containing vertex indices. 
The DOFs are not offset by `nnode`, i.e., the smallest edge DOF could be 1. 

When the input is a length 2 vector, it returns a single index for the corresponding edge DOF. 
"""
function get_edge_dof(edges::Array{Int64, 2}, mesh::Mesh)
    d = Dict{Tuple{Int64, Int64}, Int64}()
    for i = 1:mesh.nedge
        m = minimum(mesh.edges[i,:])
        M = maximum(mesh.edges[i,:])
        d[(m, M)] = i 
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
function impose_Dirichlet_boundary_conditions(A::Union{SparseMatrixCSC, Array{Float64, 2}}, 
    rhs::Array{Float64,1}, bdnode::Array{Int64, 1}, 
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

function impose_Dirichlet_boundary_conditions(A::SparseTensor, rhs::Union{Array{Float64,1}, PyObject}, 
    bdnode::Array{Int64, 1}, 
    bdval::Union{Array{Float64,1}, PyObject})
    indices = A.o.indices 
    vv = A.o.values 
    @assert size(A, 1) == size(A, 2) == length(rhs)
    @assert length(bdnode)==length(bdval)
    @assert length(bdnode)<=length(rhs)
    impose_dirichlet_ = load_op_and_grad(AdFem.libmfem,"impose_dirichlet", multiple=true)
    indices,vv,bd,rhs,bdval = convert_to_tensor(Any[indices,vv,bdnode,rhs,bdval], [Int64,Float64,Int64,Float64,Float64])
    indices, vv, rhs = impose_dirichlet_(indices,vv,bd,rhs,bdval)
    RawSparseTensor(indices, vv, size(A)...), set_shape(rhs, (size(A,2),))
end

"""
    impose_Dirichlet_boundary_conditions(A::SparseTensor, bdnode::Array{Int64, 1})

A helper function to impose homogeneous Dirichlet boundary condition. 
"""
function impose_Dirichlet_boundary_conditions(A::SparseTensor, bdnode::Array{Int64, 1})
    rhs = zeros(size(A, 1))
    bdval = zeros(length(bdnode))
    B, _ = impose_Dirichlet_boundary_conditions(A, rhs, bdnode, bdval)
    B
end


"""
    fem_to_gauss_points(u::PyObject, mesh::Mesh)
"""
function fem_to_gauss_points(u::PyObject, mesh::Mesh)
    fem_to_gauss_points_mfem_ = load_op_and_grad(AdFem.libmfem,"fem_to_gauss_points_mfem")
    u = convert_to_tensor(Any[u], [Float64]); u = u[1]
    out = fem_to_gauss_points_mfem_(u)
    set_shape(out, (get_ngauss(mesh),))
end

"""
    fem_to_gauss_points(u::Array{Float64,1}, mesh::Mesh)
"""
function fem_to_gauss_points(u::Array{Float64,1}, mesh::Mesh)
    out = zeros(get_ngauss(mesh))
    @eval ccall((:FemToGaussPointsMfem_Julia, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), 
                $out, $u)
    return out
end


"""
    dof_to_gauss_points(u::PyObject, mesh::Mesh)
    dof_to_gauss_points(u::Array{Float64,1}, mesh::Mesh)

Similar to [`fem_to_gauss_points`](@ref). The only difference is that the function uses all DOFs---which means, 
for quadratic elements, the nodal values on the edges are also used. 
"""
function dof_to_gauss_points(u::PyObject, mesh::Mesh)
    @assert length(u)==mesh.ndof
    dof_to_gauss_points_mfem_ = load_op_and_grad(AdFem.libmfem,"dof_to_gauss_points_mfem")
    u = convert_to_tensor(Any[u], [Float64]); u = u[1]
    out = dof_to_gauss_points_mfem_(u)
    set_shape(out, (get_ngauss(mesh),))
end

function dof_to_gauss_points(u::Array{Float64,1}, mesh::Mesh)
    @assert length(u)==mesh.ndof
    out = zeros(get_ngauss(mesh))
    @eval ccall((:DofToGaussPointsMfem_forward_Julia, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), 
                $out, $u)
    return out
end


"""
    get_boundary_edge_orientation(bdedge::Array{Int64, 2}, mmesh::Mesh)

Returns the orientation of the edges in `bdedge`. For example, if for a boundary element `[1,2,3]`, assume `[1,2]` is the boundary edge, 
then 

```
get_boundary_edge_orientation([1 2;2 1], mmesh) = [1.0;-1.0]
```

The return values for non-boundary edges in `bdedge` is undefined. 
"""
function get_boundary_edge_orientation(bdedge::Array{Int64, 2}, mmesh::Mesh)
    edge_orientation = Dict{Tuple{Int64, Int64}, Float64}()
    elems = mmesh.elems
    bd = bcedge(mmesh)
    bd = Set([(bd[i,1], bd[i,2]) for i = 1:size(bd, 1)])
    add_to_dict = (k, i, j)->begin 
        if (elems[k, i], elems[k, j]) in bd || (elems[k, j], elems[k, i]) in bd
            edge_orientation[(elems[k, i], elems[k, j])] = 1.0
            edge_orientation[(elems[k, j], elems[k, i])] = -1.0
        end
    end
    for i = 1:mmesh.nelem
        add_to_dict(i, 1, 2)
        add_to_dict(i, 2, 3)
        add_to_dict(i, 3, 1)
    end
    sng = zeros(size(bdedge, 1))
    for i = 1:size(bdedge, 1)
        if !haskey(edge_orientation, (bdedge[i,1], bdedge[i,2]))
            error("($(bdedge[i,1]), $(bdedge[i,2])) is not a boundary edge")
        end
        sng[i] = edge_orientation[(bdedge[i,1], bdedge[i,2])]
    end
    sng 
end



@doc raw"""
    compute_pml_term(u::Union{Array{Float64,1}, PyObject},βprime::Union{Array{Float64,1}, PyObject},
    c::Union{Array{Float64,1}, PyObject},nv::Union{Array{Float64,2}, PyObject}, mmesh::Mesh)


- `u`: a vector of length `mmesh.ndof`
- `βprime`: a vector of length $n_{\text{gauss}}$
- `c`: a tensor of size $n_{\text{gauss}}$
- `nv`: a matrix of size $n_{\text{gauss}}\times 2$

This function returns four outputs

$$\begin{aligned}k_1&=(c^2 n\partial_n u, n\partial_n\delta u)\\k_2&=(\beta'n\cdot(c^2 n\partial_n u), \delta u)\\ k_3&=(c^2\nabla^\parallel u, n\partial_n \delta u) + (c^2 n\partial_ n u, \nabla^\parallel \delta u)\\ k_4&= (c^2 \nabla^\parallel u, \nabla^\parallel \delta u)\end{aligned}$$
"""
function compute_pml_term(u::Union{Array{Float64,1}, PyObject},βprime::Union{Array{Float64,1}, PyObject},
        c::Union{Array{Float64,1}, PyObject},nv::Union{Array{Float64,2}, PyObject}, mmesh::Mesh)
    @assert length(u)==mmesh.ndof
    @assert length(βprime)==length(c)==size(nv,1)==get_ngauss(mmesh)
    compute_pml_term_ = load_op_and_grad(AdFem.libmfem,"compute_pml_term", multiple=true)
    u,betap,c,nv = convert_to_tensor(Any[u,βprime,c,nv], [Float64,Float64,Float64,Float64])
    out = compute_pml_term_(u,betap,c,nv)
    set_shape(out[1], (mmesh.ndof, )), set_shape(out[2], (mmesh.ndof, )), set_shape(out[3], (mmesh.ndof, )), set_shape(out[4], (mmesh.ndof, ))
end


@doc raw"""
    compute_pml_term(u::Union{Array{Float64,1}, PyObject},βprime::Union{Array{Float64,1}, PyObject},
    λ::Union{Array{Float64,1}, PyObject},μ::Union{Array{Float64,1}, PyObject}, nv::Union{Array{Float64,2}, PyObject}, mmesh::Mesh)


- `u`: a vector of length `mmesh.ndof`
- `βprime`: a vector of length $n_{\text{gauss}}$
- `λ`, `μ`: Lam\'e parameters
- `nv`: a matrix of size $n_{\text{gauss}}\times 2$

This is a 2D version of [`compute_pml_term`](@ref).
"""
function compute_pml_term(u::Union{Array{Float64,1}, PyObject},βprime::Union{Array{Float64,1}, PyObject},
    λ::Union{Array{Float64,1}, PyObject},μ::Union{Array{Float64,1}, PyObject}, nv::Union{Array{Float64,2}, PyObject}, mmesh::Mesh)
    @assert length(u)==2mmesh.ndof
    @assert length(βprime)==length(λ)==length(μ)==size(nv,1)==get_ngauss(mmesh)
    compute_pml_elastic_term_ = load_op_and_grad(AdFem.libmfem,"compute_pml_elastic_term", multiple=true)
    u,betap,λ,μ,nv = convert_to_tensor(Any[u,βprime,λ,μ,nv], [Float64,Float64,Float64,Float64,Float64])
    out = compute_pml_elastic_term_(u,betap,λ,μ,nv)
    set_shape(out[1], (2mmesh.ndof, )), set_shape(out[2], (2mmesh.ndof, )), set_shape(out[3], (2mmesh.ndof, )), set_shape(out[4], (2mmesh.ndof, ))
end


@doc raw"""
    solve_slip_law(v, ψ, dc, v0, a, b, f0, Δt::Float64)

Solves one step of the slip law equation 

$$\dot \psi = - \frac{|V|}{d_C}\left( a \sinh^{-1}\left( \frac{|V|}{2V_0}e^{\frac{\psi}{a}} \right) - f_0 + (b-a)*\log \frac{|V|}{V_0} \right)$$

We discretize the equation with a central difference scheme 

$$\frac{\psi^{n+1} - \psi^{n-1}}{2\Delta t} =  - \frac{|V|}{d_c}\left( a \sinh^{-1} \left( \frac{|V|}{2V_0} e^{\frac{\psi^{n+1} + \psi^{n-1}}{2a}{}} \right) - f_0 + (b-a) \log \frac{|V|}{V_0} \right)$$

- `dc`, `v0`, `a`, `b`, and `f0` are scalars 
"""
function solve_slip_law(v, ψ, dc, v0, a, b, f0, Δt::Float64)
    @assert size(v0)==size(a)==size(b)==size(f0)==size(dc)==()
    @assert size(v)==size(ψ)
    @assert length(size(v))==length(size(ψ))==1
    v, ψ, dc, v0, a, b, f0 = convert_to_tensor(Any[v, ψ, dc, v0, a, b, f0], [Float64, Float64, Float64, Float64, Float64, Float64, Float64])
    v = abs(v)
    A = (-Δt*v)/dc
    B = v/2v0*exp(ψ/2a)
    C = (ψ + 2Δt*v/dc*(f0 - (b-a)*log(v/v0)))/2a 
    solve_slip_law(A, B, C, ψ)
end


@doc raw"""
    solve_slip_law( 
        A::Union{Array{Float64,1}, PyObject}, 
        B::Union{Array{Float64,1}, PyObject}, 
        C::Union{Array{Float64,1}, PyObject},
        X0::Union{Array{Float64,1}, PyObject})

A helper function for [`solve_slip_law`](@ref) the nonlinear equation 

$$x - A\sinh^{-1}(Bx) - C = 0$$

`A`, `B`, and `C` are vectors of the same length. `X0` is the initial guess. 
"""
function solve_slip_law( 
        A::Union{Array{Float64,1}, PyObject}, 
        B::Union{Array{Float64,1}, PyObject}, 
        C::Union{Array{Float64,1}, PyObject},
        X0::Union{Array{Float64,1}, PyObject})
    @assert length(X0) == length(A) == length(B) == length(C)
    solve_slip_law_ = load_op_and_grad(AdFem.libmfem,"solve_slip_law")
    a,b,c,xinit = convert_to_tensor(Any[A, B, C, X0], [Float64,Float64,Float64,Float64])
    out = solve_slip_law_(a,b,c,xinit)
    set_shape(out, (length(a),))
end
