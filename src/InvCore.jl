function fem_impose_coupled_Dirichlet_boundary_condition(A::SparseTensor, bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    op = load_op_and_grad("$(@__DIR__)/../deps/DirichletBD/build/libDirichletBd", "dirichlet_bd", multiple=true)
    ii, jj, vv = find(A)
    ii,jj,vv,bd,m_,n_,h = convert_to_tensor([ii,jj,vv,bd,m,n,h], [Int64,Int64,Float64,Int32,Int32,Int32,Float64])
    ii1,jj1,vv1, ii2,jj2,vv2 = op(ii,jj,vv,bd,m_,n_,h)
    SparseTensor(ii1,jj1,vv1,2(m+1)*(n+1)+m*n,2(m+1)*(n+1)+m*n), SparseTensor(ii2,jj2,vv2,2(m+1)*(n+1)+m*n,2length(bd))
end

export fem_impose_Dirichlet_boundary_condition_experimental
function fem_impose_Dirichlet_boundary_condition_experimental(A::Union{SparseMatrixCSC,SparseTensor}, 
        bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)
    isa(A, SparseMatrixCSC) && (A = constant(A))
    op = load_op_and_grad("$(@__DIR__)/../deps/DirichletBD/build/libDirichletBd", "dirichlet_bd", multiple=true)
    ii, jj, vv = find(A)
    ii,jj,vv,bd,m_,n_,h = convert_to_tensor([ii,jj,vv,bdnode,m,n,h], [Int64,Int64,Float64,Int32,Int32,Int32,Float64])
    ii1,jj1,vv1, ii2,jj2,vv2 = op(ii,jj,vv,bd,m_,n_,h)
    SparseTensor(ii1,jj1,vv1,2(m+1)*(n+1),2(m+1)*(n+1)), SparseTensor(ii2,jj2,vv2,2(m+1)*(n+1),2length(bdnode))
end

function fem_impose_Dirichlet_boundary_condition(L, bdnode, m, n, h)
    idx = [bdnode; bdnode .+ (m+1)*(n+1)]
    Lbd = L[:, idx]
    L = scatter_update(L, :, idx, spzero(2*(m+1)*(n+1), length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), 2*(m+1)*(n+1)))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end

@doc raw"""
    compute_fem_stiffness_matrix1(hmat::PyObject, m::Int64, n::Int64, h::Float64)
"""
function compute_fem_stiffness_matrix1(hmat::PyObject, m::Int64, n::Int64, h::Float64)
    if length(size(hmat))!=3
        error("Only 4mn x 2 x 2 matrix `hmat` is supported.")
    end
    univariate_fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/FemStiffness1/build/libUnivariateFemStiffness","univariate_fem_stiffness", multiple=true)
    hmat,m_,n_,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    ii, jj, vv = univariate_fem_stiffness_(hmat,m_,n_,h)
    SparseTensor(ii, jj, vv, (m+1)*(n+1), (m+1)*(n+1))
end

@doc raw"""
    compute_fem_stiffness_matrix(hmat::PyObject,m::Int64, n::Int64, h::Float64)

A differentiable kernel. `hmat` has one of the following sizes 
- $3\times 3$
- $4mn \times 3 \times 3$ 
"""
function compute_fem_stiffness_matrix(hmat::PyObject, m::Int64, n::Int64, h::Float64)
    if length(size(hmat))==2
        compute_fem_stiffness_matrix2(hmat, m, n, h)
    elseif length(size(hmat))==3
        compute_fem_stiffness_matrix3(hmat, m, n, h)
    else 
        error("size hmat not valid")
    end
end

function compute_fem_stiffness_matrix2(hmat::PyObject, m::Int64, n::Int64, h::Float64)
    fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/FemStiffness/build/libFemStiffness","fem_stiffness", multiple=true)
    hmat,m_,n_,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    ii, jj, vv = fem_stiffness_(hmat,m_,n_,h)
    SparseTensor(ii, jj, vv, 2(m+1)*(n+1), 2(m+1)*(n+1))
    # ii, jj, vv
end

function compute_fem_stiffness_matrix3(hmat::PyObject,m::Int64, n::Int64, h::Float64)
    spatial_fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/SpatialFemStiffness/build/libSpatialFemStiffness",
                                    "spatial_fem_stiffness", multiple=true)
    hmat,m_,n_,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    ii, jj, vv = spatial_fem_stiffness_(hmat,m_,n_,h)
    SparseTensor(ii, jj, vv, 2(m+1)*(n+1), 2(m+1)*(n+1))
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



export eval_strain_on_gauss_pts1
"""
    eval_strain_on_gauss_pts1(u::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function eval_strain_on_gauss_pts1(u::PyObject, m::Int64, n::Int64, h::Float64)
    strain_op_univariate_ = load_op_and_grad("$(@__DIR__)/../deps/Strain1/build/libStrainOpUnivariate","strain_op_univariate")
    u,m_,n_,h = convert_to_tensor([u,m,n,h], [Float64,Int32,Int32,Float64])
    out = strain_op_univariate_(u,m_,n_,h)
    out.set_shape((4*m*n, 2))
    out 
end


export rate_state_friction
@doc raw"""
    rate_state_friction(a::Union{PyObject, Array{Float64, 1}},
    v0::Union{PyObject, Float64},psi::Union{PyObject, Array{Float64, 1}},
    sigma::Union{PyObject, Array{Float64, 1}},
    tau::Union{PyObject, Array{Float64, 1}},eta::Union{PyObject, Float64})

Computes $x = u_3(x_1, x_2)$ from rate and state friction. The governing equation is 
```math 
a \sinh^{-1}\left( \frac{x - u}{\Delta t} \frac{1}{2V_0} e^{\frac{\Psi}{a}} \right) \sigma - \tau + \eta \frac{x-u}{\Delta t} = 0
```
"""
function rate_state_friction(a::Union{PyObject, Array{Float64, 1}},
    v0::Union{PyObject, Float64},psi::Union{PyObject, Array{Float64, 1}},
    sigma::Union{PyObject, Array{Float64, 1}},
    tau::Union{PyObject, Array{Float64, 1}},eta::Union{PyObject, Float64})
    deltat = 0.0
    uold = zeros(length(a))
    rate_state_friction_ = load_op_and_grad("$(@__DIR__)/../deps/RateStateFriction/build/libRateStateFriction","rate_state_friction")
    a,uold,v0,psi,sigma,tau,eta,deltat = convert_to_tensor([a,uold,v0,psi,sigma,tau,eta,deltat], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64])
    rate_state_friction_(a,uold,v0,psi,sigma,tau,eta,deltat)
end
