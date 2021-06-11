export eval_f_on_dof_pts, get_bdedge_integration_pts, 
    gauss_weights, compute_fem_boundary_mass_matrix1, compute_fem_boundary_mass_term1,
    eval_scalar_on_boundary_edge, eval_strain_on_boundary_edge

"""
    eval_f_on_gauss_pts(f::Function, mesh::Mesh; tensor_input::Bool = false)
"""
function eval_f_on_gauss_pts(f::Function, mesh::Mesh; tensor_input::Bool = false)
    xy = gauss_nodes(mesh)
    if tensor_input 
        return f(constant(xy[:,1]), constant(xy[:,2]))
    end
    f.(xy[:,1], xy[:,2])
end


"""
    eval_f_on_fem_pts(f::Function, mesh::Mesh; tensor_input::Bool = false)
"""
function eval_f_on_fem_pts(f::Function, mesh::Mesh; tensor_input::Bool = false)
    xy = mesh.nodes
    if tensor_input 
        return f(constant(xy[:,1]), constant(xy[:,2]))
    end
    f.(xy[:,1], xy[:,2])
end

"""
    eval_f_on_fvm_pts(f::Function, mesh::Mesh; tensor_input::Bool = false)
"""
function eval_f_on_fvm_pts(f::Function, mesh::Mesh; tensor_input::Bool = false)
    xy = fvm_nodes(mesh)
    if tensor_input 
        return f(constant(xy[:,1]), constant(xy[:,2]))
    end
    f.(xy[:,1], xy[:,2])
end

"""
    eval_f_on_dof_pts(f::Function, mesh::Mesh)

Evaluates `f` on the DOF points. 

- For P1 element, the DOF points are FEM points and therefore `eval_f_on_dof_pts` is equivalent to `eval_on_on_fem_pts`.
- For P2 element, the DOF points are FEM points plus the middle point for each edge. 

Returns a vector of length `mesh.ndof`.
"""
function eval_f_on_dof_pts(f::Function, mesh::Mesh)
    if size(mesh.conn, 2)==3
        return eval_f_on_fem_pts(f, mesh)
    end
    xy = mesh.nodes 
    xy2 = zeros(mesh.nedge, 2)
    for i = 1:mesh.nedge
        xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
    end
    xy = [xy;xy2]
    f.(xy[:,1], xy[:,2])
end

"""
    compute_fem_source_term1(f::PyObject, mesh::Mesh)
"""
function compute_fem_source_term1(f::PyObject, mesh::Mesh)
    @assert length(f)==get_ngauss(mesh)
    fem_source_scalar_ = load_op_and_grad(libmfem,"fem_source_scalar")
    f = convert_to_tensor(Any[f], [Float64]); f = f[1]
    out = fem_source_scalar_(f)
    n = mesh.ndof
    set_shape(out, (n,))
end

"""
    compute_fem_source_term1(f::Array{Float64,1}, mesh::Mesh)
"""
function compute_fem_source_term1(f::Array{Float64,1}, mesh::Mesh)
    @assert length(f)==get_ngauss(mesh)
    out = zeros(mesh.ndof)
    @eval ccall((:FemSourceScalar_forward_Julia, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), $out, $f)
    out
end

"""
    compute_fem_source_term(f1::Union{PyObject,Array{Float64,2}}, f2::Union{PyObject,Array{Float64,2}}, mesh::Mesh)
"""
function compute_fem_source_term(f1::Union{PyObject,Array{Float64,1}}, f2::Union{PyObject,Array{Float64,1}}, mesh::Mesh)
    [compute_fem_source_term1(f1, mesh); compute_fem_source_term1(f2, mesh)]
end

function compute_fem_source_term(f::Union{PyObject,Array{Float64,1}},  mmesh::Mesh)
    @assert length(f)==2*get_ngauss(mmesh)
    compute_fem_source_term(f[1:get_ngauss(mmesh)], f[get_ngauss(mmesh)+1:end], mmesh)
end


"""
    compute_fem_laplace_matrix1(kappa::PyObject, mesh::Mesh)
"""
function compute_fem_laplace_matrix1(kappa::PyObject, mesh::Mesh)
    @assert length(kappa) == get_ngauss(mesh)
    fem_laplace_scalar_ = load_op_and_grad(AdFem.libmfem,"fem_laplace_scalar", multiple=true)
    kappa = convert_to_tensor(Any[kappa], [Float64]); kappa = kappa[1]
    indices, vv = fem_laplace_scalar_(kappa)
    n = mesh.ndof
    RawSparseTensor(indices, vv, n, n)
end

"""
    compute_fem_laplace_matrix1(kappa::Array{Float64,1}, mesh::Mesh)
"""
function compute_fem_laplace_matrix1(kappa::Array{Float64,1}, mesh::Mesh)
    @assert length(kappa) == get_ngauss(mesh)
    N = get_ngauss(mesh) * size(mesh.conn, 2)^2
    indices = zeros(Int64, 2N)
    vv = zeros(N)
    @eval ccall((:FemLaplaceScalar_forward_Julia, $LIBMFEM), Cvoid, (Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}), $indices, $vv, $kappa)
    indices = reshape(indices, 2, N)'|>Array
    sparse(indices[:,1] .+ 1, indices[:,2] .+ 1, vv, mesh.ndof, mesh.ndof)
end

"""
    compute_fem_laplace_matrix(kappa::Union{PyObject, Array{Float64, 1}}, mesh::Mesh)
"""
function compute_fem_laplace_matrix(kappa::Union{PyObject, Array{Float64, 1}}, mesh::Mesh)
    Z = compute_fem_laplace_matrix1(kappa, mesh)
    if isa(Z, SparseMatrixCSC)
        [Z spzeros(mesh.ndof, mesh.ndof)
        spzeros(mesh.ndof, mesh.ndof) Z]
    else
        [Z spzero(mesh.ndof) 
        spzero(mesh.ndof) Z]
    end
end


compute_fem_laplace_matrix1(mesh::Mesh) = compute_fem_laplace_matrix1(ones(get_ngauss(mesh)), mesh)
compute_fem_laplace_matrix(mesh::Mesh) = compute_fem_laplace_matrix(ones(get_ngauss(mesh)), mesh)


"""
    fem_impose_Dirichlet_boundary_condition1(L::SparseTensor, bdnode::Array{Int64}, mesh::Mesh)

A differentiable kernel for imposing the Dirichlet boundary of a scalar-valued function. 
"""
function fem_impose_Dirichlet_boundary_condition1(L::SparseTensor, bdnode::Array{Int64}, mesh::Mesh)
    @warn "Consider imposing boundary conditions using an algebraic approach: impose_Dirichlet_boundary_conditions" maxlog=1
    idx = bdnode
    Lbd = L[:, idx]
    L = scatter_update(L, :, idx, spzero(size(mesh.nodes, 1), length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), size(mesh.nodes, 1)))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end


"""
    compute_interaction_term(p::Union{PyObject,Array{Float64, 1}}, mesh::Mesh)
"""
function compute_interaction_term(p::Union{PyObject,Array{Float64, 1}}, mesh::Mesh)
    compute_interaction_term_mfem_ = load_op_and_grad(AdFem.libmfem,"compute_interaction_term_mfem")
    p = convert_to_tensor(Any[p], [Float64]); p = p[1]
    out = compute_interaction_term_mfem_(p)
    set_shape(out, (2mesh.ndof, ))
end

"""
    compute_fem_mass_matrix1(rho::Union{PyObject, Array{Float64, 1}}, mesh::Mesh)
"""
function compute_fem_mass_matrix1(rho::Union{PyObject, Array{Float64, 1}}, mesh::Mesh)
    compute_fem_mass_matrix_mfem_ = load_op_and_grad(AdFem.libmfem,"compute_fem_mass_matrix_mfem", multiple=true)
    rho = convert_to_tensor(Any[rho], [Float64]); rho = rho[1]
    indices, vals = compute_fem_mass_matrix_mfem_(rho)
    n = mesh.ndof
    A = RawSparseTensor(indices, vals, n, n)
    A
end

function compute_fem_mass_matrix1(mmesh::Mesh)
    compute_fem_mass_matrix1(ones(get_ngauss(mmesh)), mmesh)
end

# function compute_fem_mass_matrix(mmesh::Mesh)
# end

# function compute_fem_mass_matrix(rho::Union{PyObject, Array{Float64, 1}}, mmesh::Mesh)
# end

"""
    compute_fem_advection_matrix1(u::Union{Array{Float64,1}, PyObject},v::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
    compute_fem_advection_matrix1(u::Array{Float64,1}, v::Array{Float64,1}, mesh::Mesh)
"""
function compute_fem_advection_matrix1(u::Union{Array{Float64,1}, PyObject},v::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
    @assert length(u)==length(v)==get_ngauss(mesh)
    compute_fem_advection_matrix_mfem_ = load_op_and_grad(AdFem.libmfem,"compute_fem_advection_matrix_mfem", multiple=true)
    u,v = convert_to_tensor(Any[u,v], [Float64,Float64])
    indices, vals = compute_fem_advection_matrix_mfem_(u,v)
    n = mesh.ndof
    RawSparseTensor(indices, vals, n, n)
end

function compute_fem_advection_matrix1(u::Array{Float64,1}, v::Array{Float64,1}, mesh::Mesh)
    @assert length(u)==length(v)==get_ngauss(mesh)
    N = get_ngauss(mesh) * mesh.elem_ndof^2
    indices = zeros(Int64, 2N)
    vv = zeros(N)
    @eval ccall((:ComputeFemAdvectionMatrixMfem_forward_Julia, $LIBMFEM), 
        Cvoid, (Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), $indices, $vv, $u, $v)
    sparse(indices[1:2:end] .+ 1, indices[2:2:end] .+ 1, vv, mesh.ndof, mesh.ndof)
end





"""
    compute_interaction_matrix(mesh::Mesh)
"""
function compute_interaction_matrix(mesh::Mesh)
    elem_dof = size(mesh.conn, 2)
    N = get_ngauss(mesh) * 2 * elem_dof
    ii = zeros(Int64, N)
    jj = zeros(Int64, N)
    vv = zeros(Float64, N)
    @eval ccall((:ComputeInteractionMatrixMfem, $LIBMFEM), Cvoid, (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}), $ii, $jj, $vv)
    m = size(mesh.elems, 1)
    sparse(ii, jj, vv, m, 2mesh.ndof)
end

"""
    eval_grad_on_gauss_pts1(u::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
"""
function eval_grad_on_gauss_pts1(u::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
    @assert length(u)==mesh.ndof
    fem_grad_mfem_ = load_op_and_grad(AdFem.libmfem,"fem_grad_mfem")
    u = convert_to_tensor(Any[u], [Float64]); u = u[1]
    out = fem_grad_mfem_(u)
    m = size(gauss_nodes(mesh), 1)
    set_shape(out, (m, 2))
end

"""
    eval_grad_on_gauss_pts(u::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
"""
# function eval_grad_on_gauss_pts(u::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
#     n = size(mesh.nodes, 1)
#     m = size(gauss_nodes(mesh), 1)
#     r1 = eval_grad_on_gauss_pts1(u[1:n], mesh)
#     r2 = eval_grad_on_gauss_pts1(u[n+1:end], mesh)
#     out = zeros(m, 2, 2)
#     for i = 1:m
#         out[i, 1, 1] = r1[i, 1]  # out is numerical array but r1 is tensor, cannot assign value
#         out[i, 1, 2] = r1[i, 2]
#         out[i, 2, 1] = r2[i, 1] 
#         out[i, 2, 2] = r2[i, 2] 
#     end
#     return out 
# end

@doc raw"""
    compute_fem_stiffness_matrix(kappa::PyObject, mesh::Mesh)
    compute_fem_stiffness_matrix(kappa::Array{Float64, 3}, mesh::Mesh)
    compute_fem_stiffness_matrix(kappa::Array{Float64, 2}, mesh::Mesh)

Computes the stiffness matrix. Here, the acceptable sizes of $\kappa$ 
- a $3\times 3$ matrix, which is the pointwise uniform stiffness matrix 
- a $N_g \times 3 \times 3$ tensor, which includes a specific stiffness matrix at each Gauss node
"""
function compute_fem_stiffness_matrix(kappa::PyObject, mesh::Mesh)
    if length(size(kappa))==2
        kappa = reshape(repeat(reshape(kappa, (-1,)), get_ngauss(mesh)), (get_ngauss(mesh), 3, 3))
    end
    @assert size(kappa) == (get_ngauss(mesh), 3, 3)
    compute_fem_stiffness_matrix_mfem_ = load_op_and_grad(AdFem.libmfem,"compute_fem_stiffness_matrix_mfem", multiple=true)
    kappa = convert_to_tensor(Any[kappa], [Float64]); kappa = kappa[1]
    kappa = reshape(kappa, (-1,))
    indices, vv = compute_fem_stiffness_matrix_mfem_(kappa)
    RawSparseTensor(indices, vv, 2mesh.ndof, 2mesh.ndof)
end

function compute_fem_stiffness_matrix(kappa::Array{Float64, 2}, mesh::Mesh)
    @assert size(kappa) == (3, 3)
    K = zeros(get_ngauss(mesh), 3, 3)
    for i = 1:get_ngauss(mesh)
        K[i, :, :] = kappa 
    end
    compute_fem_stiffness_matrix(K, mesh)
end

function compute_fem_stiffness_matrix(kappa::Array{Float64, 3}, mesh::Mesh)
    @assert size(kappa) == (get_ngauss(mesh), 3, 3)
    K = zeros(length(kappa))
    s = 1
    for i = 1:size(kappa, 1)
        for p = 1:3
            for q = 1:3
                K[s] = kappa[i, p, q]
                s += 1
            end
        end
    end
    N = (size(mesh.conn, 2) * 2)^2 * get_ngauss(mesh);
    indices = zeros(Int64, 2N)
    vv = zeros(N)
    @eval ccall((:ComputeFemStiffnessMatrixMfem_forward_Julia, $LIBMFEM), Cvoid, 
            (Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}), $indices, $vv, $K)
    indices = reshape(indices, 2, N)'|>Array 
    sparse(indices[:,1] .+ 1, indices[:, 2] .+ 1, vv, 2mesh.ndof, 2mesh.ndof)
end

@doc raw"""
    get_bdedge_integration_pts(bdedge::Array{Int64, 2}, mesh::Mesh)

Returns Gauss quadrature points on the boundary edge as a $n\times 2$ matrix. Here $n$ is the number of rows for `bdedge`.
"""
function get_bdedge_integration_pts(bdedge::Array{Int64, 2}, mmesh::Mesh)
    order = mmesh.lorder
    bdnode_x, bdnode_y = _traction_get_nodes(bdedge, mmesh)
    bdnode_x = bdnode_x'[:]
    bdnode_y = bdnode_y'[:]
    bdN = size(bdedge, 1)
    ngauss = Int64(@eval ccall((:ComputeFemTractionTermMfem_forward_getNGauss, $LIBMFEM), Cint, 
            (Cint,), Int32($order)))
    x = zeros(ngauss * bdN)
    y = zeros(ngauss * bdN)
    @eval ccall((:ComputeFemTractionTermMfem_forward_getGaussPoints, $LIBMFEM), Cvoid, 
    (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint), $x, $y, 
        $bdnode_x, $bdnode_y, Int32($bdN), Int32($order))
    [x y]
end



@doc raw"""
    _traction_get_nodes(bdedge::Array{Int64, 2}, mesh::Mesh)

Returns $x$ and $y$ coordinates for each element that contains `bdedge`.

For example, if `bdedge = [2,4]`, this function finds the corresponding element, e.g., one that contains vertices
2, 4, 5. Then two arrays are returned: 
- `node_x`: $x$ coordinates of node 2, 4, and 5
- `node_y`: $y$ coordinates of node 2, 4, and 5
"""
function _traction_get_nodes(bdedge::Array{Int64, 2}, mesh::Mesh)
    nodes_x = zeros(size(bdedge, 1), 3)
    nodes_y = zeros(size(bdedge, 1), 3)
    _edge_to_elem_map = Dict{Tuple{Int64, Int64}, Int64}()
    dic = (x,y)->(minimum([x,y]), maximum([x,y]))
    add_dic = (e1, e2, e3)->begin 
        if haskey(_edge_to_elem_map, dic(e1, e2))
            delete!(_edge_to_elem_map, dic(e1, e2))
        else
            _edge_to_elem_map[dic(e1, e2)] = e3
        end
    end
    for i = 1:mesh.nelem
        e1, e2, e3 = mesh.elems[i,:]
        add_dic(e1, e2, e3)
        add_dic(e3, e2, e1)
        add_dic(e1, e3, e2)
    end
    for i = 1:size(bdedge, 1)
        idx = _edge_to_elem_map[dic(bdedge[i,1], bdedge[i,2])]
        nodes_x[i, :] = [mesh.nodes[bdedge[i,1], 1]; mesh.nodes[bdedge[i,2], 1]; mesh.nodes[idx, 1]]
        nodes_y[i, :] = [mesh.nodes[bdedge[i,1], 2]; mesh.nodes[bdedge[i,2], 2]; mesh.nodes[idx, 2]]
    end
    nodes_x, nodes_y
end

@doc raw"""
    bcedge(mesh::Mesh)

Returns all boundary edges as a set of integer pairs (edge vertices).
"""
function bcedge(mesh::Mesh)
    _edge_to_elem_map = Set{Tuple{Int64, Int64}}()
    dic = (x,y)->(minimum([x,y]), maximum([x,y]))
    add_dic = (e1, e2)->begin 
        if dic(e1, e2) in _edge_to_elem_map
            delete!(_edge_to_elem_map, dic(e1, e2))
        else
            push!(_edge_to_elem_map, dic(e1, e2))
        end
    end
    for i = 1:mesh.nelem
        e1, e2, e3 = mesh.elems[i,:]
        add_dic(e1, e2)
        add_dic(e3, e2)
        add_dic(e1, e3)
    end
    out = zeros(Int64, length(_edge_to_elem_map), 2)
    for (k,s) in enumerate(_edge_to_elem_map)
        out[k,:] = [s[1] s[2]]
    end
    out
end

@doc raw"""
    bcedge(f::Function, mesh::Mesh)

Returns all edge indices that satisfies `f(x1, y1, x2, y2) = true`
Here the edge endpoints are given by $(x_1, y_1)$ and $(x_2, y_2)$.
"""
function bcedge(f::Function, mmesh::Mesh)
    out = bcedge(mmesh)
    edges = []
    for i = 1:size(out, 1)
        e1, e2 = out[i,:]
        x1, y1 = mmesh.nodes[e1, :]
        x2, y2 = mmesh.nodes[e2, :]
        if f(x1, y1, x2, y2)
            push!(edges, [e1 e2])
        end
    end
    vcat(edges...)
end


@doc raw"""
    bcnode(mesh::Mesh; by_dof::Bool = true)

Returns all boundary node indices. 

If `by_dof = true`, `bcnode` returns the global indices for boundary DOFs. 

- For `P2` elements, the returned values are boundary node DOFs + boundary edge DOFs (offseted by `mesh.nnode`)
- For `BDM1` elements, the returned values are boundary edge DOFs + boundary edge DOFs offseted by `mesh.nedge`
"""
function bcnode(mmesh::Mesh; by_dof::Bool = true)
    bdedge = bcedge(mmesh)
    if by_dof && mmesh.elem_type == P2
        edgedof = get_edge_dof(bdedge, mmesh) .+ mmesh.nnode
        [collect(Set(bdedge[:])); edgedof]
    elseif by_dof && mmesh.elem_type == BDM1
        bd = get_edge_dof(bdedge, mmesh) 
        [bd; bd .+ mmesh.nedge]
    else
        collect(Set(bdedge[:]))
    end
end

"""
    bcnode(f::Function, mesh::Mesh; by_dof::Bool = true)

Returns the boundary node DOFs that satisfies `f(x,y) = true`.


!!! note

    For BDM1 element and `by_dof = true`, because the degrees of freedoms are associated with edges, `f` has the signature

    ```julia 
    f(x1::Float64, y1::Float64, x2::Float64, y2::Float64)::Bool
    ```

    `bcnode` only returns DOFs on edges such that `f(x1, y1, x2, y2)=true`. 
"""
function bcnode(f::Function, mesh::Mesh; by_dof::Bool = true)
    if mesh.elem_type==BDM1
        return _bcnode_bdm1(f, mesh; by_dof=by_dof)
    end
    @assert mesh.elem_type in [P1, P2]
    nd = bcnode(mesh, by_dof=by_dof)
    out = Int64[]
    for i = 1:length(nd)
        if nd[i]<=mesh.nnode
            if f(mesh.nodes[nd[i], 1], mesh.nodes[nd[i],2])
                push!(out, nd[i])
            end
        else
            a, b = mesh.edges[nd[i]-mesh.nnode, :]
            xy = (mesh.nodes[a, :] + mesh.nodes[b, :])/2
            if f(xy[1], xy[2])
                push!(out, nd[i])
            end
        end
    end
    out
end

function _bcnode_bdm1(f::Function, mmesh::Mesh; by_dof::Bool = true)
    if !by_dof
        return bcnode(mmesh, by_dof = false)
    end 
    bdedge = bcedge((x1, y1, x2, y2)->f(x1, y1, x2, y2), mmesh)
    if length(bdedge)==0
        return Int64[]
    end
    e = get_edge_dof(bdedge, mmesh)
    [e; e.+mmesh.nedge]
end

@doc raw"""
    compute_fem_traction_term1(t::Array{Float64, 1},
    bdedge::Array{Int64,2}, mesh::Mesh)

Computes the boundary integral 

$$\int_{\Gamma} t(x, y) \delta u dx$$

Returns a vector of size `dof`.
"""
function compute_fem_traction_term1(t::Array{Float64, 1},
            bdedge::Array{Int64,2}, mesh::Mesh)
    # sort bdedge so that bdedge[i,1] < bdedge[i,2]
    for i = 1:size(bdedge, 1)
        if bdedge[i,1]>bdedge[i,2]
            bdedge[i,:] = [bdedge[i,2]; bdedge[i,1]]
        end
    end
    order = mesh.lorder
    D = _edge_dict(mesh)
    node_x = zeros(size(bdedge, 1))
    node_y = zeros(size(bdedge, 1))
    ngauss = Int64(@eval ccall((:ComputeFemTractionTermMfem_forward_getNGauss, $LIBMFEM), Cint, 
            (Cint,), Int32($order)))
    bdN = size(bdedge, 1)
    @assert length(t) == ngauss * bdN
    if mesh.elem_type==P1
        dofs = zeros(Int64, 2bdN)
        for i = 1:bdN 
            dofs[2*i-1] = bdedge[i,1] - 1
            dofs[2*i] = bdedge[i,2] - 1
        end
    elseif mesh.elem_type==P2
        dofs = zeros(Int64, 3bdN)
        for i = 1:bdN 
            dofs[3*i-2] = bdedge[i,1] - 1
            dofs[3*i-1] = bdedge[i,2] - 1
            dofs[3*i] = D[(bdedge[i,1], bdedge[i,2])] + mesh.nnode - 1
        end
    elseif mesh.elem_type==BDM1
        dofs = zeros(Int64, 2bdN)
        for i = 1:bdN 
            e = D[(bdedge[i,1], bdedge[i,2])]
            dofs[2*i-1] = e - 1
            dofs[2*i] = e - 1 + mesh.nedge
        end
    end
    out = zeros(mesh.ndof)
    bdnode_x, bdnode_y = _traction_get_nodes(bdedge, mesh)
    bdnode_x = bdnode_x'[:]
    bdnode_y = bdnode_y'[:]
    if mesh.elem_type in [P1, P2]
        @eval ccall((:ComputeFemTractionTermMfem_forward_Julia, $LIBMFEM), Cvoid, 
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint), 
                $out, $t, Int32.($dofs), $bdnode_x, $bdnode_y, Int32($bdN), Int32($order))
    elseif mesh.elem_type == BDM1
        sn = get_boundary_edge_orientation(bdedge, mesh)
        @eval ccall((:ComputeBDMTractionTermMfem_forward_Julia, $LIBMFEM), Cvoid, 
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
            $out, $sn, $t, Int32.($dofs), $bdnode_x, $bdnode_y, Int32($bdN), Int32($order))
    end
    return out 
end

"""
    compute_fem_traction_term(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, mesh::Mesh)
"""
function compute_fem_traction_term(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, mesh::Mesh)
    @assert size(t,2)==2
    [compute_fem_traction_term1(t[:,1], bdedge, mesh);
    compute_fem_traction_term1(t[:,2], bdedge, mesh)]
end

"""
    compute_fem_traction_term(t1::Array{Float64, 1}, t2::Array{Float64, 1},
    bdedge::Array{Int64,2}, mesh::Mesh)
"""
function compute_fem_traction_term(t1::Array{Float64, 1}, t2::Array{Float64, 1},
    bdedge::Array{Int64,2}, mesh::Mesh)
    [compute_fem_traction_term1(t1, bdedge, mesh);
    compute_fem_traction_term1(t2, bdedge, mesh)]
end


"""
    eval_f_on_boundary_edge(f::Function, bdedge::Array{Int64, 2}, mesh::Mesh; tensor_input::Bool = false)

Evaluates `f` on the boundary **Gauss points**. Here `f` has the signature

```f(Float64, Float64)::Float64```

or 

```f(PyObject, PyObject)::PyObject```
"""
function eval_f_on_boundary_edge(f::Function, bdedge::Array{Int64, 2}, 
        mesh::Mesh; tensor_input::Bool = false)
    pts = get_bdedge_integration_pts(bdedge, mesh)
    if tensor_input
        f(constant(pts[:,1]), constant(pts[:,2]))
    else
        f.(pts[:,1], pts[:,2])
    end
end


"""
    eval_f_on_boundary_node(f::Function, bdnode::Array{Int64}, mesh::Mesh)
"""
function eval_f_on_boundary_node(f::Function, bdnode::Array{Int64}, mesh::Mesh)
    out = zeros(length(bdnode))
    for i = 1:length(bdnode)
        if bdnode[i]>mesh.nnode 
            a, b = mesh.edges[bdnode[i]-mesh.nnode, :]
            xy = (mesh.nodes[a, :] + mesh.nodes[b, :])/2
            out[i] = f(xy[1], xy[2])
        else
            a = bdnode[i]
            out[i] = f(mesh.nodes[a, 1], mesh.nodes[a, 2])
        end
    end
    out
end


"""
    compute_von_mises_stress_term(K::Array{Float64, 3}, u::Array{Float64, 1}, mesh::Mesh)
    compute_von_mises_stress_term(K::Array{Float64, 2}, u::Array{Float64, 1}, mesh::Mesh)
"""
function compute_von_mises_stress_term(K::Array{Float64, 3}, u::Array{Float64, 1}, mesh::Mesh)
    @assert length(u) == 2mesh.ndof
    @assert size(K) == (get_ngauss(mesh), 3, 3)
    hmat = zeros(9*get_ngauss(mesh))
    for i = 1:get_ngauss(mesh)
        for j = 1:3
            for k = 1:3
                hmat[(i-1)*9+(j-1)*3+k] = K[i, j, k]
            end
        end
    end
    out = zeros(get_ngauss(mesh))
    @eval ccall((:mfem_compute_von_mises_stress_term, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), 
                $out, $hmat, $u)
    out
end

function compute_von_mises_stress_term(K::Array{Float64, 2}, u::Array{Float64, 1}, mesh::Mesh)
    Ks = zeros(get_ngauss(mesh), 3, 3)
    for i = 1:get_ngauss(mesh)
        Ks[i,:,:] = K 
    end
    compute_von_mises_stress_term(Ks, u, mesh)
end

"""
    compute_fem_laplace_term1(u::Array{Float64, 1},nu::Array{Float64, 1}, mesh::Mesh)
"""
function compute_fem_laplace_term1(u::Array{Float64, 1},nu::Array{Float64, 1}, mesh::Mesh)
    @assert length(u) == mesh.ndof
    @assert length(nu) == get_ngauss(mesh)
    out = zeros(mesh.ndof)
    @eval ccall((:ComputeLaplaceTermMfem_forward_Julia, $LIBMFEM), Cvoid, 
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), $out, $nu, $u)
    out
end

"""
    compute_fem_laplace_term1(u::Union{PyObject, Array{Float64, 1}},
                                nu::Union{PyObject, Array{Float64, 1}},
                                mesh::Mesh)
"""
function compute_fem_laplace_term1(u::Union{PyObject, Array{Float64, 1}},
                                   nu::Union{PyObject, Array{Float64, 1}},
                                   mesh::Mesh)
    @assert length(u) == mesh.ndof
    @assert length(nu) == get_ngauss(mesh)
    compute_laplace_term_mfem_ = load_op_and_grad(AdFem.libmfem,"compute_laplace_term_mfem")
    u,nu = convert_to_tensor(Any[u,nu], [Float64,Float64])
    out = compute_laplace_term_mfem_(u,nu)
    set_shape(out, (mesh.ndof,))
end

"""
    gauss_weights(mmesh::Mesh)

Returns the weights for each Gauss points.
"""
function gauss_weights(mmesh::Mesh)
    w = zeros(get_ngauss(mmesh))
    @eval ccall((:mfem_get_gauss_weights, $LIBMFEM), Cvoid, (Ptr{Cdouble},), $w)
    w
end


"""
    compute_fvm_source_term(f::Array{Float64, 1}, mmesh::Mesh)
"""
function compute_fvm_source_term(f::Array{Float64, 1}, mmesh::Mesh)
    w = gauss_weights(mmesh)
    src = zeros(mmesh.nelem)
    ngauss_per_elem = get_ngauss(mmesh)÷mmesh.nelem
    for i = 1:mmesh.nelem
        idx = (i-1)*ngauss_per_elem+1:i*ngauss_per_elem
        src[i] = sum(f[idx].*w[idx])
    end
    src
end


@doc raw"""
    compute_strain_energy_term(Sigma::Array{Float64, 2}, mmesh::Mesh)

Computes the strain energy term 

$$\int_A \sigma : \varepsilon (\delta u) dx$$

Here $\sigma$ is a fourth-order tensor. `Sigma` is a `ngauss × 3` matrix, each row represents  $[\sigma_{11}, \sigma_{22}, \sigma_{12}]$ at 
each Gauss point. 

The output is a length `2mmesh.ndof` vector. 
"""
function compute_strain_energy_term(Sigma::Array{Float64, 2}, mmesh::Mesh)
    @assert size(Sigma,1)==get_ngauss(mmesh)
    @assert size(Sigma,2)==3
    out = zeros(2mmesh.ndof)
    @eval ccall((:ComputeStrainEnergyTermMfem_forward_Julia, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), $out, $(Array(Sigma')))
    out
end


"""
    compute_strain_energy_term(Sigma::PyObject, mmesh::Mesh)
"""
function compute_strain_energy_term(Sigma::PyObject, mmesh::Mesh)
    @assert size(Sigma,1)==get_ngauss(mmesh)
    @assert size(Sigma,2)==3
    compute_strain_energy_term_mfem_ =  @eval load_op_and_grad($libmfem,"compute_strain_energy_term_mfem")
    sigma = convert_to_tensor(Any[Sigma], [Float64]); sigma = sigma[1]
    se = compute_strain_energy_term_mfem_(sigma)
    set_shape(se, (2mmesh.ndof,))
end



"""
    eval_strain_on_gauss_pts(u::Array{Float64}, mmesh::Mesh)

Evaluates the strain on Gauss points. `u` is a vector of size `2mmesh.ndof`.

The output is a `ngauss × 3` vector.
"""
function eval_strain_on_gauss_pts(u::Array{Float64}, mmesh::Mesh)
    @assert length(u)==2mmesh.ndof
    ε = zeros(3get_ngauss(mmesh))
    @eval ccall((:EvalStrainOnGaussPts_forward_Julia, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), $ε, $u)
    Array(reshape(ε, 3, get_ngauss(mmesh))')
end

"""
    eval_strain_on_gauss_pts(u::PyObject, mmesh::Mesh)
"""
function eval_strain_on_gauss_pts(u::PyObject, mmesh::Mesh)
    @assert length(u)==2mmesh.ndof
    eval_strain_on_gauss_pts_ = @eval load_op_and_grad($libmfem,"eval_strain_on_gauss_pts")
    u = convert_to_tensor(Any[u], [Float64]); u = u[1]
    ε = eval_strain_on_gauss_pts_(u)
    set_shape(ε, (get_ngauss(mmesh), 3))
end

@doc raw"""
    compute_fem_boundary_mass_matrix1(c::Union{Array{Float64}, PyObject}, bdedge::Array{Int64, 2}, mmesh::Mesh)

Computes the matrix 

$$\int_\Gamma cu \delta u ds$$

The parameters are 
- `bdedge`: a $N_e \times 2$ integer array, the boundary edge to integrate on
- `c`: given by a vector of length $4N_e$; currently, each edge has 4 quadrature points;

The output is a $N_v\times N_v$ sparse matrix. 
"""
function compute_fem_boundary_mass_matrix1(c::Union{Array{Float64}, PyObject}, bdedge::Array{Int64, 2}, mmesh::Mesh)
    @assert length(c)==4*size(bdedge, 1)
    compute_boundary_mass_matrix_one_ = @eval load_op_and_grad($libmfem,"compute_boundary_mass_matrix_one", multiple=true)
    c,idx = convert_to_tensor(Any[c,bdedge], [Float64,Int64])
    ij, vv = compute_boundary_mass_matrix_one_(c,idx)
    RawSparseTensor(ij, vv, mmesh.ndof, mmesh.ndof)
end


@doc raw"""
    compute_fem_boundary_mass_term1(u::Union{Array{Float64}, PyObject}, 
        c::Union{Array{Float64}, PyObject}, bdedge::Array{Int64, 2}, mmesh::Mesh)

Computes the term 

$$\int_\Gamma cu \delta u ds$$

The parameters are 
- `u` : a vector of length $N_v$ 
- `bdedge` : a $N_e \times 2$ integer array, the boundary edge to integrate on
- `c`: given by a vector of length $4N_e$; currently, each edge has 4 quadrature points;

The output is a $N_v\times N_v$ sparse matrix. 
"""
function compute_fem_boundary_mass_term1(u::Union{Array{Float64}, PyObject}, 
        c::Union{Array{Float64}, PyObject}, bdedge::Array{Int64, 2}, mmesh::Mesh)
    @assert length(u)==mmesh.nnode
    compute_fem_boundary_mass_matrix1(c, bdedge, mmesh) * u 
end


@doc raw"""
    eval_scalar_on_boundary_edge(u::Union{PyObject, Array{Float64, 1}},
            edge::Array{Int64, 1}, mmesh::Mesh)

Returns an array of values on the Gauss quadrature nodes for each edge. 

- `u`: nodal values 
- `edge`: a $N\times 2$ integer array; each row represents an edge $(x_1, x_2)$

The returned array consists of $(y_1, y_2, \ldots)$

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/AdFem/docs/eval_scalar_on_boundary_edge.png?raw=true)
"""
function eval_scalar_on_boundary_edge(u::Union{PyObject, Array{Float64, 1}},
        edge::Array{Int64, 2}, mmesh::Mesh)
    eval_scalar_on_boundary_edge_ = @eval load_op_and_grad($libmfem,"eval_scalar_on_boundary_edge")
    u,edge = convert_to_tensor(Any[u,edge], [Float64,Int64])
    out = eval_scalar_on_boundary_edge_(u,edge)
    N = @eval ccall((:get_LineIntegralN, $(AdFem.LIBMFEM)), Cint, ())
    reshape(out, (N*size(edge, 1), ))
end

@doc raw"""
    eval_strain_on_boundary_edge(u::Union{PyObject, Array{Float64, 1}},
    edge::Array{Int64, 2}, mmesh::Mesh)

Returns an array of strain tensors on the Gauss quadrature nodes for each edge. 

The returned value has size $N_e N_g\times 3$. Here $N_e$ is the number of edges, and $N_g$ is the number of 
Gauss points on the edge. Each row of `edge`, $(l,r)$, has the following property: $l < r$
"""
function eval_strain_on_boundary_edge(u::Union{PyObject, Array{Float64, 1}},
    edge::Array{Int64, 2}, mmesh::Mesh)
    @assert length(u) == 2mmesh.ndof
    eval_strain_on_boundary_edge_ = @eval load_op_and_grad($libmfem,"eval_strain_on_boundary_edge")
    u,edge = convert_to_tensor(Any[u,edge], [Float64,Int64])
    out = eval_strain_on_boundary_edge_(u,edge)
    N = @eval ccall((:get_LineIntegralN, $(AdFem.LIBMFEM)), Cint, ())
    reshape(out, (N*size(edge, 1), 3))
end
