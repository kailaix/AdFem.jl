function fem_impose_coupled_Dirichlet_boundary_condition(A::SparseTensor, bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    op = load_op_and_grad("$(@__DIR__)/../deps/DirichletBD/build/libDirichletBd", "dirichlet_bd", multiple=true)
    ii, jj, vv = find(A)
    ii,jj,vv,bd,m_,n_,h = convert_to_tensor([ii,jj,vv,bd,m,n,h], [Int64,Int64,Float64,Int32,Int32,Int32,Float64])
    ii1,jj1,vv1, ii2,jj2,vv2 = op(ii,jj,vv,bd,m_,n_,h)
    SparseTensor(ii1,jj1,vv1,2(m+1)*(n+1)+m*n,2(m+1)*(n+1)+m*n), SparseTensor(ii2,jj2,vv2,2(m+1)*(n+1)+m*n,2length(bd))
end

function fem_impose_Dirichlet_boundary_condition(L, bdnode, m, n, h)
    idx = [bdnode; bdnode .+ (m+1)*(n+1)]
    Lbd = L[:, idx]
    L = scatter_update(L, :, idx, spzero(2*(m+1)*(n+1), length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), 2*(m+1)*(n+1)))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end

"""
    compute_fem_stiffness_matrix(hmat::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel
"""
function compute_fem_stiffness_matrix(hmat::PyObject, m::Int64, n::Int64, h::Float64)
    fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/FemStiffness/build/libFemStiffness","fem_stiffness", multiple=true)
    hmat,m_,n_,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    ii, jj, vv = fem_stiffness_(hmat,m_,n_,h)
    SparseTensor(ii, jj, vv, 2(m+1)*(n+1), 2(m+1)*(n+1))
    # ii, jj, vv
end

"""
    compute_strain_energy_term(S::PyObject,m::Int64, n::Int64, h::Float64)

A differentiable kernel. 
"""
function compute_strain_energy_term(S::PyObject,m::Int64, n::Int64, h::Float64)
    strain_energy_ = load_op_and_grad("$(@__DIR__)/../deps/StrainEnergy/build/libStrainEnergy","strain_energy")
    sigma,m_,n_,h = convert_to_tensor([S,m,n,h], [Float64,Int32,Int32,Float64])
    out = strain_energy_(sigma,m_,n_,h)
    out.set_shape((2*(m+1)*(n+1),))
    out
end

"""
    eval_strain_on_gauss_pts(u::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function eval_strain_on_gauss_pts(u::PyObject, m::Int64, n::Int64, h::Float64)
    strain_op_ = load_op_and_grad("$(@__DIR__)/../deps/Strain/build/libStrainOp","strain_op")
    u,m_,n_,h = convert_to_tensor([u,m,n,h], [Float64,Int32,Int32,Float64])
    out = strain_op_(u,m_,n_,h)
    out.set_shape((4*m*n, 3))
    out 
end