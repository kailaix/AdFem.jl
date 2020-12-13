export pcl_impose_Dirichlet_boundary_conditions, pcl_compute_fem_laplace_matrix1
@doc raw"""
    pcl_impose_Dirichlet_boundary_conditions(indices::Array{Int64, 2}, bdnode::Array{Int64, 1}, outdof::Int64)

Computes the Jacobian matrix for [`impose_Dirichlet_boundary_conditions`](@ref). Assume that `impose_Dirichlet_boundary_conditions`
transforms a sparse matrix `A` to `B`, and `v_A` and `v_B` are the nonzero entries, this function computes 

$$J_{ij} = \frac{\partial (v_B)_j}{\partial (v_A)_i}$$

- `indices`: the **1-based** $n_A\times 2$ index array for `A`;
- `bdnode`: boundary DOF (the same as inputs to `impose_Dirichlet_boundary_conditions`)
- `outdof`: the number of nonzero entries in `B`, i.e., length of `v_B`
"""
function pcl_impose_Dirichlet_boundary_conditions(indices::Array{Int64, 2}, bdnode::Array{Int64, 1}, outdof::Int64)
    bdN = Int32(length(bdnode))
    sN = Int32(size(indices, 1))
    J = zeros(sN, outdof)
    @eval ccall((:pcl_ImposeDirichlet, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Clonglong}, Ptr{Clonglong}, Cint, Cint),
        $J, $indices, $bdnode, $bdN, $sN)
    return J
end

@doc raw"""
    pcl_compute_fem_laplace_matrix1(mmesh::Mesh)

Computes the Jacobian matrix for [`pcl_compute_fem_laplace_matrix1`](@ref). Assume we contruct a sparse matrix via 
```julia
A = compute_fem_laplace_matrix1(κ, mmesh)
v = A.o.values 
```
Then the function returns 

$$J = \frac{\partial v_j}{\partial κ_i}$$
"""
function pcl_compute_fem_laplace_matrix1(mmesh::Mesh)
    J = zeros(get_ngauss(mmesh), mmesh.elem_ndof^2 * get_ngauss(mmesh))
    @eval ccall((:pcl_FemLaplaceScalar_Jacobian, $LIBMFEM), Cvoid, (Ptr{Cdouble},),$J)
    J
end
