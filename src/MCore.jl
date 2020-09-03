function eval_f_on_gauss_pts(f::Function, mesh::Mesh)
    xy = gauss_nodes(mesh)
    f.(xy[:,1], xy[:,2])
end

function eval_f_on_fem_pts(f::Function, mesh::Mesh)
    xy = mesh.nodes
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