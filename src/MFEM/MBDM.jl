export compute_fem_div_bdm_matrix, compute_fem_bdm_mass_matrix

@doc raw"""
    compute_fem_div_bdm_matrix(mmesh::Mesh) 

Computes the coefficient matrix for 

$$\int_\Omega \text{div} \tau \delta u dx$$

Here $\tau \in \mathbb{R}^{2\times 2}$ is a fourth-order tensor (not necessarily symmetric). `mmesh` 
uses the BDM1 finite element. The output is a `mmesh.nelem × 2mmesh.nedge` matrix. 
"""
function compute_fem_div_bdm_matrix(mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    ii = zeros(Int64, get_ngauss(mmesh) * mmesh.elem_ndof)
    jj = zeros(Int64, get_ngauss(mmesh) * mmesh.elem_ndof)
    jj = zeros(get_ngauss(mmesh) * mmesh.elem_ndof)
    @eval ccall((:BDMDiveMatrixMfem, $LIBMFEM), Cvoid, (Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Cdouble}), 
        $ii, $jj, $vv)
    sparse(ii, jj, vv, mmesh.nelem, 2mmesh.nedge)
end

@doc raw"""
    compute_fem_bdm_mass_matrix(alpha::Union{Array{Float64,1}, PyObject},beta::Union{Array{Float64,1}, PyObject}, mmesh::Mesh)

Computes 

$$\int_\Omega A\sigma : \delta \tau dx$$

Here 

$$A\sigma = \alpha \sigma + \beta \text{tr} \sigma I$$

$\sigma$ and $\tau$ are both fourth-order tensors. The output is a  `2mmesh.nedge × 2mmesh.nedge` matrix.
"""
function compute_fem_bdm_mass_matrix(alpha::Union{Array{Float64,1}, PyObject},beta::Union{Array{Float64,1}, PyObject}, mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    bdm_inner_product_matrix_mfem_ = load_op_and_grad(libmfem,"bdm_inner_product_matrix_mfem", multiple=true)
    alpha,beta = convert_to_tensor(Any[alpha,beta], [Float64,Float64])
    indices, vv = bdm_inner_product_matrix_mfem_(alpha,beta)
    RawSparseTensor(indices, values, mmesh.ndof, mmesh.ndof)
end

"""
    compute_fem_bdm_mass_matrix(alpha::Array{Float64,1},beta::Array{Float64,1}, mmesh::Mesh)
"""
function compute_fem_bdm_mass_matrix(alpha::Array{Float64,1},beta::Array{Float64,1}, mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    @assert length(alpha)==length(beta)==get_ngauss(mmesh)
    N = mmesh.elem_ndof * get_ngauss(mmesh) * 6;
    indices = zeros(Int64, 2N)
    vv = zeros(Int64, N)
    @eval ccall((:BDMInnerProductMatrixMfem_forward_Julia, $LIBMFEM), 
        Cvoid, (Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), $indices, $vv, $alpha, $beta)
    sparse(indices[1:2:end].+1, indices[2:2:end].+1, vv, 2mmesh.nedge, 2mmesh.nedge)
end


