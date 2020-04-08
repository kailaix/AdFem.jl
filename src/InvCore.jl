export compute_strain_energy_term1, compute_space_varying_tangent_elasticity_matrix

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


"""
    fem_impose_Dirichlet_boundary_condition1(L::SparseTensor, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)

A differentiable kernel for imposing the Dirichlet boundary of a scalar-valued function. 
"""
function fem_impose_Dirichlet_boundary_condition1(L::SparseTensor, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)
    idx = bdnode
    Lbd = L[:, idx]
    L = scatter_update(L, :, idx, spzero((m+1)*(n+1), length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), (m+1)*(n+1)))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end

"""
    fem_impose_Dirichlet_boundary_condition(L::SparseTensor, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)

A differentiable kernel for imposing the Dirichlet boundary of a vector-valued function. 
"""
function fem_impose_Dirichlet_boundary_condition(L, bdnode, m, n, h)
    idx = [bdnode; bdnode .+ (m+1)*(n+1)]
    Lbd = L[:, idx]
    Lbd = scatter_update(Lbd, idx, :, spzero(length(idx), length(idx)))
    L = scatter_update(L, :, idx, spzero(2*(m+1)*(n+1), length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), 2*(m+1)*(n+1)))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end

@doc raw"""
    compute_fem_stiffness_matrix1(hmat::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel for computing the stiffness matrix. 
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

@doc raw"""
    compute_strain_energy_term1(sigma::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable  operator.
"""
function compute_strain_energy_term1(sigma::PyObject, m::Int64, n::Int64, h::Float64)
    strain_energy_univariate_ = load_op_and_grad("$(@__DIR__)/../deps/StrainEnergy1/build/libStrainEnergyUnivariate","strain_energy_univariate")
    sigma,m_,n_,h = convert_to_tensor([sigma,m,n,h], [Float64,Int32,Int32,Float64])
    out = strain_energy_univariate_(sigma,m_,n_,h)
    set_shape(out, ((m+1)*(n+1),))
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


export compute_vel
@doc raw"""
    compute_vel(a::Union{PyObject, Array{Float64, 1}},
    v0::Union{PyObject, Float64},psi::Union{PyObject, Array{Float64, 1}},
    sigma::Union{PyObject, Array{Float64, 1}},
    tau::Union{PyObject, Array{Float64, 1}},eta::Union{PyObject, Float64})

Computes $x = u_3(x_1, x_2)$ from rate and state friction. The governing equation is 
```math 
a \sinh^{-1}\left( \frac{x - u}{\Delta t} \frac{1}{2V_0} e^{\frac{\Psi}{a}} \right) \sigma - \tau + \eta \frac{x-u}{\Delta t} = 0
```
"""
function compute_vel(a::Union{PyObject, Array{Float64, 1}},
    v0::Union{PyObject, Float64},psi::Union{PyObject, Array{Float64, 1}},
    sigma::Union{PyObject, Array{Float64, 1}},
    tau::Union{PyObject, Array{Float64, 1}},eta::Union{PyObject, Float64})
    compute_vel_ = load_op_and_grad("$(@__DIR__)/../deps/ComputeVel/build/libComputeVel","compute_vel")
    a,v0,psi,sigma,tau,eta = convert_to_tensor([a,v0,psi,sigma,tau,eta], [Float64,Float64,Float64,Float64,Float64,Float64])
    compute_vel_(a,v0,psi,sigma,tau,eta)
end

@doc raw"""
    compute_space_varying_tangent_elasticity_matrix(mu::Union{PyObject, Array{Float64,1}},m::Int64,n::Int64,h::Float64,type::Int64=1)

Computes the space varying tangent elasticity matrix given $\mu$. It returns a matrix of size $4mn\times 2\times 2$

* If `type==1`, the $i$-th matrix will be 

$$\begin{bmatrix}\mu_i & 0 \\ 0 & \mu_i \end{bmatrix}$$

* If `type==2`, the $i$-th matrix will be 

$$\begin{bmatrix}\mu_i & 0 \\ 0 & \mu_{i+4mn} \end{bmatrix}$$

* If `type==3`, the $i$-th matrix will be 

$$\begin{bmatrix}\mu_i & \mu_{i+8mn} \\ \mu_{i+8mn} & \mu_{i+4mn}\end{bmatrix}$$
"""
function compute_space_varying_tangent_elasticity_matrix(mu::Union{PyObject, Array{Float64,1}},m::Int64,n::Int64,h::Float64,type::Int64=1)
    spatial_varying_tangent_elastic_ = load_op_and_grad("$(@__DIR__)/../deps/SpatialVaryingTangentElastic/build/libSpatialVaryingTangentElastic","spatial_varying_tangent_elastic")
    mu,m_,n_,h,type = convert_to_tensor([mu,m,n,h,type], [Float64,Int64,Int64,Float64,Int64])
    H = spatial_varying_tangent_elastic_(mu,m_,n_,h,type)
    set_shape(H, (4*m*n, 2, 2))
end