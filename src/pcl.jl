export pcl_impose_Dirichlet_boundary_conditions, pcl_compute_fem_laplace_matrix1
"""
    pcl_impose_Dirichlet_boundary_conditions
"""
function pcl_impose_Dirichlet_boundary_conditions(indices, bdnode, dof)
    bdN = length(bdnode)
    sN = size(indices, 1)
    J = zeros(sN, dof)
    @eval ccall((:pcl_ImposeDirichlet, $LIBADFEM), Cvoid, (Ptr{Cdouble}, Ptr{Clonglong}, Ptr{Clonglong}, Cint, Cint),
        $J, $indices, $bdnode, $bdN, $sN)
    return J
end

function pcl_compute_fem_laplace_matrix1(mmesh)
    J = zeros(get_ngauss(mmesh), mmesh.elem_ndof^2 * get_ngauss(mmesh))
    @eval ccall((:pcl_FemLaplaceScalar_Jacobian, $LIBMFEM), Cvoid, (Ptr{Cdouble},),$J)
    J
end
