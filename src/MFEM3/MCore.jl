"""
    compute_fem_laplace_term1(u::Union{PyObject, Array{Float64, 1}},
        nu::Union{PyObject, Array{Float64, 1}},
        mesh::Mesh3)
"""
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

########################################## 

"""
    compute_fem_laplace_matrix1(kappa::PyObject, mesh::Mesh3)
"""
function compute_fem_laplace_matrix1(kappa::PyObject, mesh::Mesh3)
    @assert length(kappa) == get_ngauss(mesh)
    fem_laplace_scalar_ = load_op_and_grad(AdFem.libmfem3,"fem_laplace_scalar_t", multiple=true)
    kappa = convert_to_tensor(Any[kappa], [Float64]); kappa = kappa[1]
    indices, vv = fem_laplace_scalar_(kappa)
    n = mesh.ndof
    RawSparseTensor(indices, vv, n, n)
end

"""
    compute_fem_laplace_matrix1(kappa::Array{Float64,1}, mesh::Mesh3)
"""
function compute_fem_laplace_matrix1(kappa::Array{Float64,1}, mesh::Mesh3)
    @assert length(kappa) == get_ngauss(mesh)
    N = get_ngauss(mesh) * size(mesh.conn, 2)^2
    indices = zeros(Int64, 2N)
    vv = zeros(N)
    @eval ccall((:FemLaplaceScalarT_forward_Julia, $LIBMFEM3), Cvoid, (Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}), $indices, $vv, $kappa)
    indices = reshape(indices, 2, N)'|>Array
    sparse(indices[:,1] .+ 1, indices[:,2] .+ 1, vv, mesh.ndof, mesh.ndof)
end

"""
    compute_fem_laplace_matrix(kappa::Union{PyObject, Array{Float64, 1}}, mesh::Mesh3)
"""
function compute_fem_laplace_matrix(kappa::Union{PyObject, Array{Float64, 1}}, mesh::Mesh3)
    Z = compute_fem_laplace_matrix1(kappa, mesh)
    if isa(Z, SparseMatrixCSC)
        [Z spzeros(mesh.ndof, mesh.ndof)
        spzeros(mesh.ndof, mesh.ndof) Z]
    else
        [Z spzero(mesh.ndof) 
        spzero(mesh.ndof) Z]
    end
end

compute_fem_laplace_matrix1(mesh::Mesh3) = compute_fem_laplace_matrix1(ones(get_ngauss(mesh)), mesh)
compute_fem_laplace_matrix(mesh::Mesh3) = compute_fem_laplace_matrix(ones(get_ngauss(mesh)), mesh)
##########################################



"""
    compute_fem_source_term1(f::PyObject, mesh::Mesh3)
"""
function compute_fem_source_term1(f::PyObject, mesh::Mesh3)
    @assert length(f)==get_ngauss(mesh)
    fem_source_scalar_ = load_op_and_grad(libmfem3,"fem_source_scalar_t")
    f = convert_to_tensor(Any[f], [Float64]); f = f[1]
    out = fem_source_scalar_(f)
    n = mesh.ndof
    set_shape(out, (n,))
end

"""
    compute_fem_source_term1(f::Array{Float64,1}, mesh::Mesh3)
"""
function compute_fem_source_term1(f::Array{Float64,1}, mesh::Mesh3)
    @assert length(f)==get_ngauss(mesh)
    out = zeros(mesh.ndof)
    @eval ccall((:FemSourceScalarT_forward_Julia, $LIBMFEM3), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), $out, $f)
    out
end

"""
    compute_fem_source_term(f1::Union{PyObject,Array{Float64,2}}, 
        f2::Union{PyObject,Array{Float64,2}}, mesh::Mesh3)
"""
function compute_fem_source_term(f1::Union{PyObject,Array{Float64,1}}, f2::Union{PyObject,Array{Float64,1}}, mesh::Mesh3)
    [compute_fem_source_term1(f1, mesh); compute_fem_source_term1(f2, mesh)]
end



##########################################

"""
    compute_fem_mass_matrix1(ρ::Union{PyObject, Array{Float64, 1}}, 
            mmesh::Mesh3)
"""
function compute_fem_mass_matrix1(ρ::Union{PyObject, Array{Float64, 1}}, 
        mmesh::Mesh3)
        compute_fem_mass_matrix_mfem_t_ = load_op_and_grad(libmfem3,"compute_fem_mass_matrix_mfem_t", multiple=true)
        rho = convert_to_tensor(Any[ρ], [Float64]); rho = rho[1]
        indices, vv = compute_fem_mass_matrix_mfem_t_(rho)
        RawSparseTensor(indices, vv, mmesh.ndof, mmesh.ndof)
end

"""
    compute_fem_mass_matrix1(mmesh::Mesh3)
"""
compute_fem_mass_matrix1(mmesh::Mesh3) = compute_fem_mass_matrix1(ones(get_ngauss(mmesh)), mmesh)
