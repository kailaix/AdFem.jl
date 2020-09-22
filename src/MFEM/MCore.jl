function eval_f_on_gauss_pts(f::Function, mesh::Mesh)
    xy = gauss_nodes(mesh)
    f.(xy[:,1], xy[:,2])
end

function eval_f_on_fem_pts(f::Function, mesh::Mesh)
    xy = mesh.nodes
    f.(xy[:,1], xy[:,2])
end

function eval_f_on_fvm_pts(f::Function, mesh::Mesh)
    xy = fvm_nodes(mesh)
    f.(xy[:,1], xy[:,2])
end

"""
    compute_fem_source_term1(f::PyObject, mesh::Mesh)
"""
function compute_fem_source_term1(f::PyObject, mesh::Mesh)
    fem_source_scalar_ = load_op_and_grad(libmfem,"fem_source_scalar")
    f = convert_to_tensor(Any[f], [Float64]); f = f[1]
    out = fem_source_scalar_(f)
    n = size(mesh.nodes, 1)
    set_shape(out, (n,))
end

"""
    compute_fem_laplace_matrix1(kappa::PyObject, mesh::Mesh)
"""
function compute_fem_laplace_matrix1(kappa::PyObject, mesh::Mesh)
    fem_laplace_scalar_ = load_op_and_grad(libmfem,"fem_laplace_scalar", multiple=true)
    kappa = convert_to_tensor(Any[kappa], [Float64]); kappa = kappa[1]
    indices, vv = fem_laplace_scalar_(kappa)
    n = size(mesh.nodes, 1)
    RawSparseTensor(indices, vv, n, n)
end


"""
    fem_impose_Dirichlet_boundary_condition1(L::SparseTensor, bdnode::Array{Int64}, mesh::Mesh)

A differentiable kernel for imposing the Dirichlet boundary of a scalar-valued function. 
"""
function fem_impose_Dirichlet_boundary_condition1(L::SparseTensor, bdnode::Array{Int64}, mesh::Mesh)
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
    compute_interaction_term_mfem_ = load_op_and_grad(PoreFlow.libmfem,"compute_interaction_term_mfem")
    p = convert_to_tensor(Any[p], [Float64]); p = p[1]
    out = compute_interaction_term_mfem_(p)
    set_shape(out, (2*size(mesh.nodes, 1), ))
end

"""
    compute_fem_mass_matrix1(rho::Union{PyObject, Array{Float64, 1}}, mesh::Mesh)
"""
function compute_fem_mass_matrix1(rho::Union{PyObject, Array{Float64, 1}}, mesh::Mesh)
    compute_fem_mass_matrix_mfem_ = load_op_and_grad(PoreFlow.libmfem,"compute_fem_mass_matrix_mfem", multiple=true)
    rho = convert_to_tensor(Any[rho], [Float64]); rho = rho[1]
    indices, vals = compute_fem_mass_matrix_mfem_(rho)
    n = size(mesh.nodes, 1)
    A = RawSparseTensor(indices, vals, n, n)
    A
end

"""
    compute_fem_advection_matrix1(u::Union{Array{Float64,1}, PyObject},v::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
"""
function compute_fem_advection_matrix1(u::Union{Array{Float64,1}, PyObject},v::Union{Array{Float64,1}, PyObject}, mesh::Mesh)
    compute_fem_advection_matrix_mfem_ = load_op_and_grad(PoreFlow.libmfem,"compute_fem_advection_matrix_mfem", multiple=true)
    u,v = convert_to_tensor(Any[u,v], [Float64,Float64])
    indices, vals = compute_fem_advection_matrix_mfem_(u,v)
    n = size(mesh.nodes, 1)
    RawSparseTensor(indices, vals, n, n)
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
    fem_grad_mfem_ = load_op_and_grad(PoreFlow.libmfem,"fem_grad_mfem")
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