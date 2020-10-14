export compute_fem_bdm_div_matrix1, compute_fem_bdm_div_matrix, 
        compute_fem_bdm_mass_matrix, compute_fem_bdm_mass_matrix1,
        compute_fem_bdm_skew_matrix, impose_bdm_traction_boundary_condition1,
        impose_bdm_traction_boundary_condition

@doc raw"""
    compute_fem_bdm_div_matrix1(mmesh::Mesh) 

Computes the coefficient matrix for 

$$\int_\Omega \text{div} \tau \delta u dx$$

Here $\tau \in \mathbb{R}^{2}$ is a second-order tensor (not necessarily symmetric). `mmesh` 
uses the BDM1 finite element. The output is a `mmesh.nelem × 2mmesh.nedge` matrix. 
"""
function compute_fem_bdm_div_matrix1(mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    ii = zeros(Int64, get_ngauss(mmesh) * mmesh.elem_ndof)
    jj = zeros(Int64, get_ngauss(mmesh) * mmesh.elem_ndof)
    vv = zeros(get_ngauss(mmesh) * mmesh.elem_ndof)
    @eval ccall((:BDMDiveMatrixMfem, $LIBMFEM), Cvoid, (Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Cdouble}), 
        $ii, $jj, $vv)
    sparse(ii, jj, vv, mmesh.nelem, 2mmesh.nedge)
end

@doc raw"""
    compute_fem_bdm_div_matrix(mmesh::Mesh) 

Computes the coefficient matrix for 

$$\int_\Omega \text{div} \tau \delta u dx$$

Here $\tau \in \mathbb{R}^{2\times 2}$ is a fourth-order tensor (not necessarily symmetric). `mmesh` 
uses the BDM1 finite element. The output is a `2mmesh.nelem × 4mmesh.nedge` matrix. 
"""
function compute_fem_bdm_div_matrix(mmesh::Mesh)
    C = compute_fem_bdm_div_matrix1(mmesh)
    [C spzeros(mmesh.nelem, 2mmesh.nedge)
    spzeros(mmesh.nelem, 2mmesh.nedge) C]
end

@doc raw"""
    compute_fem_bdm_mass_matrix(alpha::Union{Array{Float64,1}, PyObject},beta::Union{Array{Float64,1}, PyObject}, mmesh::Mesh)

Computes 

$$\int_\Omega A\sigma : \delta \tau dx$$

Here 

$$A\sigma = \alpha \sigma + \beta \text{tr} \sigma I$$

$\sigma$ and $\tau$ are both fourth-order tensors. The output is a  `4mmesh.nedge × 4mmesh.nedge` matrix.
"""
function compute_fem_bdm_mass_matrix(alpha::Union{Array{Float64,1}, PyObject},beta::Union{Array{Float64,1}, PyObject}, mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    bdm_inner_product_matrix_mfem_ = load_op_and_grad(libmfem,"bdm_inner_product_matrix_mfem", multiple=true)
    alpha,beta = convert_to_tensor(Any[alpha,beta], [Float64,Float64])
    indices, vv = bdm_inner_product_matrix_mfem_(alpha,beta)
    RawSparseTensor(indices, values, mmesh.ndof, mmesh.ndof)
end

@doc raw"""
    compute_fem_bdm_mass_matrix(mmesh::Mesh)

Same as [`compute_fem_bdm_mass_matrix`](@ref), except that 

$$$$
"""
function compute_fem_bdm_mass_matrix(mmesh::Mesh)
    C = compute_fem_bdm_mass_matrix1(mmesh)
    [C spzeros(mmesh.ndof, mmesh.ndof)
    spzeros(mmesh.ndof, mmesh.ndof) C]
end

"""
    compute_fem_bdm_mass_matrix(alpha::Array{Float64,1},beta::Array{Float64,1}, mmesh::Mesh)
"""
function compute_fem_bdm_mass_matrix(alpha::Array{Float64,1},beta::Array{Float64,1}, mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    @assert length(alpha)==length(beta)==get_ngauss(mmesh)
    N = mmesh.elem_ndof^2 * get_ngauss(mmesh) * 6;
    indices = zeros(Int64, 2N)
    vv = zeros(N)
    @eval ccall((:BDMInnerProductMatrixMfem_forward_Julia, $LIBMFEM), 
        Cvoid, (Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), $indices, $vv, $alpha, $beta)
    # @info minimum(indices[1:2:end].+1), maximum(indices[1:2:end].+1)
    # @info minimum(indices[2:2:end].+1), maximum(indices[2:2:end].+1)
    sparse(indices[1:2:end].+1, indices[2:2:end].+1, vv, 4mmesh.nedge, 4mmesh.nedge)
end

@doc raw"""
    compute_fem_bdm_mass_matrix1(alpha::Array{Float64,1}, mmesh::Mesh)

Computes 

$$\int_\Omega \alpha\sigma \cdot \delta \tau dx$$

Here $\alpha$ is a scalar, and $\sigma$ and $\delta \tau$ are second order tensors. 

The returned value is a `2mmesh.nedge × 2mmesh.nedge` matrix. 
"""
function compute_fem_bdm_mass_matrix1(alpha::Array{Float64,1}, mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    @assert length(alpha)==get_ngauss(mmesh)
    N = mmesh.elem_ndof^2 * get_ngauss(mmesh);
    indices = zeros(Int64, 2N)
    vv = zeros(N)
    @eval ccall((:BDMInnerProductMatrixMfem1_forward_Julia, $LIBMFEM), 
        Cvoid, (Ptr{Clonglong}, Ptr{Cdouble},  Ptr{Cdouble}), $indices, $vv, $alpha)
    sparse(indices[1:2:end].+1, indices[2:2:end].+1, vv, 2mmesh.nedge, 2mmesh.nedge)
end

@doc raw"""
    compute_fem_bdm_mass_matrix1(mmesh::Mesh)

Same as [`compute_fem_bdm_mass_matrix1`](@ref), except that $\alpha\equiv 1$
"""
function compute_fem_bdm_mass_matrix1(mmesh::Mesh)
    compute_fem_bdm_mass_matrix1(ones(get_ngauss(mmesh)), mmesh)
end


@doc raw"""
    compute_fem_bdm_skew_matrix(mmesh::Mesh)

Computes 
$$\int_\Omega \sigma : v dx$$
where 

$$v = \begin{bmatrix}0 & \rho \\-\rho & 0 \end{bmatrix}$$

Here $\sigma$ is a fourth-order tensor. 

The returned value is a `mmesh.nelem × 4mmesh.nedge` matrix. 
"""
function compute_fem_bdm_skew_matrix(mmesh::Mesh)
    @assert mmesh.elem_type == BDM1
    N = mmesh.elem_ndof * get_ngauss(mmesh) * 2;
    ii = zeros(Int64, N)
    jj = zeros(Int64, N)
    vv = zeros(N)
    @eval ccall((:BDMSkewSymmetricMatrixMfem, $LIBMFEM), 
        Cvoid, (Ptr{Clonglong}, Ptr{Clonglong},  Ptr{Cdouble}), $ii, $jj, $vv)
    sparse(ii, jj, vv, mmesh.nelem, 4mmesh.nedge)
end


@doc raw"""
    impose_bdm_traction_boundary_condition1(gN::Array{Float64, 1}, bdedge::Array{Int64, 2}, mesh::Mesh)

Imposes the BDM traction boundary condition 

$$\int_{\Gamma} \sigma \mathbf{n} g_N ds$$

Here $\sigma$ is a second-order tensor. `gN` is defined on the Gauss points, e.g. 

```julia 
gN = eval_f_on_boundary_edge(func, bdedge, mesh)
```
"""
function impose_bdm_traction_boundary_condition1(gN::Array{Float64, 1}, bdedge::Array{Int64, 2}, mesh::Mesh)
    @assert mesh.elem_type == BDM1
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
    @assert length(gN) == ngauss * bdN

    dofs = zeros(Int64, 2bdN)
    for i = 1:bdN 
        e = D[(bdedge[i,1], bdedge[i,2])]
        dofs[2*i-1] = e 
        dofs[2*i] = e + mesh.nedge
    end

    out = zeros(2bdN)
    bdnode_x, bdnode_y = _traction_get_nodes(bdedge, mesh)
    bdnode_x = bdnode_x'[:]
    bdnode_y = bdnode_y'[:]

    sn = get_boundary_edge_orientation(bdedge, mesh)
    @eval ccall((:ComputeBDMTractionBoundaryMfem_forward_Julia, $LIBMFEM), Cvoid, 
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
        $out, $sn, $gN, $bdnode_x, $bdnode_y, Int32($bdN), Int32($order))
    dofs, out
end


@doc raw"""
    impose_bdm_traction_boundary_condition(g1:Array{Float64, 1}, g2:Array{Float64, 1},
    bdedge::Array{Int64, 2}, mesh::Mesh)

Imposes the BDM traction boundary condition 

$$\int_{\Gamma} \sigma \mathbf{n} \cdot \mathbf{g}_N ds$$

Here $\sigma$ is a fourth-order tensor. $\mathbf{g}_N = \begin{bmatrix}g_{N1}\\ g_{N2}\end{bmatrix}$
See [`impose_bdm_traction_boundary_condition1`](@ref).

Returns a `dof` vector and a `val` vector. 
"""
function impose_bdm_traction_boundary_condition(g1::Array{Float64, 1}, g2::Array{Float64, 1},
                        bdedge::Array{Int64, 2}, mesh::Mesh)
    d1, v1 = impose_bdm_traction_boundary_condition1(g1, bdedge, mesh)
    d2, v2 = impose_bdm_traction_boundary_condition1(g2, bdedge, mesh)
    [d1;d2 .+ mesh.ndof], [v1;v2]
end