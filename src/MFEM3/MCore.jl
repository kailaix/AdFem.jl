function compute_fem_laplace_term1(u::Union{PyObject, Array{Float64, 1}},
    nu::Union{PyObject, Array{Float64, 1}},
    mesh::Mesh3)
    @assert length(u) == mesh.ndof
    @assert length(nu) == get_ngauss(mesh)
    compute_laplace_term_mfem_ = load_op_and_grad(AdFem.libmfem3,"compute_laplace_term_mfem_t")
    u,nu = convert_to_tensor(Any[u,nu], [Float64,Float64])
    out = compute_laplace_term_mfem_(u,nu)
    set_shape(out, (mesh.ndof,))
end