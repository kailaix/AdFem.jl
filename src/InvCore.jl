export compute_strain_energy_term1, compute_space_varying_tangent_elasticity_matrix,
    compute_fvm_advection_term, compute_fvm_tpfa_matrix, compute_fvm_advection_matrix,
    compute_fem_advection_matrix1, compute_fem_advection_matrix,
    compute_interaction_term

function fem_impose_coupled_Dirichlet_boundary_condition(A::SparseTensor, bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    op = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow", "dirichlet_bd", multiple=true)
    ii, jj, vv = find(A)
    ii,jj,vv,bd,m_,n_,h = convert_to_tensor([ii,jj,vv,bd,m,n,h], [Int64,Int64,Float64,Int32,Int32,Int32,Float64])
    ii1,jj1,vv1, ii2,jj2,vv2 = op(ii,jj,vv,bd,m_,n_,h)
    SparseTensor(ii1,jj1,vv1,2(m+1)*(n+1)+m*n,2(m+1)*(n+1)+m*n), SparseTensor(ii2,jj2,vv2,2(m+1)*(n+1)+m*n,2length(bd))
end

export fem_impose_Dirichlet_boundary_condition_experimental
function fem_impose_Dirichlet_boundary_condition_experimental(A::Union{SparseMatrixCSC,SparseTensor}, 
        bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)
    isa(A, SparseMatrixCSC) && (A = constant(A))
    op = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow", "dirichlet_bd", multiple=true)
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
    M, N = size(L)
    L = scatter_update(L, :, idx, spzero(M, length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), N))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end

"""
    fem_impose_Dirichlet_boundary_condition(L::SparseTensor, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)

A differentiable kernel for imposing the Dirichlet boundary of a vector-valued function. 
"""
function fem_impose_Dirichlet_boundary_condition(L::SparseTensor, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)
    M, N = size(L)
    idx = [bdnode; bdnode .+ (m+1)*(n+1)]
    Lbd = L[:, idx]
    Lbd = scatter_update(Lbd, idx, :, spzero(length(idx), length(idx)))
    L = scatter_update(L, :, idx, spzero(M, length(idx)))
    L = scatter_update(L, idx, :,  spzero(length(idx), N))
    L = scatter_update(L, idx, idx, spdiag(length(idx)))
    L, Lbd
end

@doc raw"""
    compute_fem_stiffness_matrix1(hmat::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel for computing the stiffness matrix. 
Two possible shapes for `hmat` are supported: 

- $4mn \times 2\times 2$
- $2 \times 2$
"""
function compute_fem_stiffness_matrix1(hmat::PyObject, m::Int64, n::Int64, h::Float64)
    if !(length(size(hmat)) in [2,3])
        error("Only 4mn x 2 x 2 or 2 x 2 `hmat` is supported.")
    end
    @assert size(hmat,2)==2
    univariate_fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","univariate_fem_stiffness", multiple=true)
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
        @assert size(hmat,1)==size(hmat,2)==3
        compute_fem_stiffness_matrix2(hmat, m, n, h)
    elseif length(size(hmat))==3
        @assert size(hmat,2)==size(hmat,3)==3
        compute_fem_stiffness_matrix3(hmat, m, n, h)
    else 
        error("size hmat not valid")
    end
end

function compute_fem_stiffness_matrix2(hmat::PyObject, m::Int64, n::Int64, h::Float64)
    fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","fem_stiffness", multiple=true)
    hmat,m_,n_,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    ii, jj, vv = fem_stiffness_(hmat,m_,n_,h)
    SparseTensor(ii, jj, vv, 2(m+1)*(n+1), 2(m+1)*(n+1))
    # ii, jj, vv
end

function compute_fem_stiffness_matrix3(hmat::PyObject,m::Int64, n::Int64, h::Float64)
    spatial_fem_stiffness_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow",
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
    strain_energy_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","strain_energy")
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
    strain_energy_univariate_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","strain_energy_univariate")
    sigma,m_,n_,h = convert_to_tensor([sigma,m,n,h], [Float64,Int32,Int32,Float64])
    out = strain_energy_univariate_(sigma,m_,n_,h)
    set_shape(out, ((m+1)*(n+1),))
end

"""
    eval_strain_on_gauss_pts(u::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function eval_strain_on_gauss_pts(u::PyObject, m::Int64, n::Int64, h::Float64)
    strain_op_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","strain_op")
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
    strain_op_univariate_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","strain_op_univariate")
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
    compute_vel_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","compute_vel")
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
    spatial_varying_tangent_elastic_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","spatial_varying_tangent_elastic")
    mu,m_,n_,h,type = convert_to_tensor([mu,m,n,h,type], [Float64,Int64,Int64,Float64,Int64])
    H = spatial_varying_tangent_elastic_(mu,m_,n_,h,type)
    set_shape(H, (4*m*n, 2, 2))
end


"""
    compute_fvm_tpfa_matrix(K::PyObject, bc::Array{Int64,2}, pval::Union{Array{Float64},PyObject}, m::Int64, n::Int64, h::Float64)
    
A differentiable kernel for [`compute_fvm_tpfa_matrix`](@ref). 
"""
function compute_fvm_tpfa_matrix(K::PyObject, bc::Array{Int64,2}, pval::PyObject, m::Int64, n::Int64, h::Float64)
    tpfa_op_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","tpfa_op", multiple=true)
    K,bc,pval,m_,n_,h = convert_to_tensor(Any[K,bc,pval,m,n,h], [Float64,Int64,Float64,Int64,Int64,Float64])
    ii, jj, vv, rhs = tpfa_op_(K,bc,pval,m_,n_,h)
    SparseTensor(ii + 1, jj + 1, vv, m*n, m*n), set_shape(rhs, m*n) 
end

compute_fvm_tpfa_matrix(K::Array{Float64,1}, bc::Array{Int64,2}, 
                        pval::PyObject, m::Int64, n::Int64, h::Float64) =
                                    compute_fvm_tpfa_matrix(constant(K), bc, pval, m, n, h)


compute_fvm_tpfa_matrix(K::PyObject, bc::Array{Int64,2}, 
    pval::Array{Float64,1}, m::Int64, n::Int64, h::Float64) =
            compute_fvm_tpfa_matrix(K, bc, constant(pval), m, n, h)


"""
    compute_fvm_tpfa_matrix(K::PyObject, m::Int64, n::Int64, h::Float64) 

A differentiable kernel for [`compute_fvm_tpfa_matrix`](@ref). 
"""
function compute_fvm_tpfa_matrix(K::PyObject, m::Int64, n::Int64, h::Float64) 
    bc = zeros(Int64, 0, 2)
    pval = zeros(Float64, 0)
    A, _ = compute_fvm_tpfa_matrix(K, bc, pval, m, n, h)
    A 
end
        
        

@doc raw"""
    compute_fvm_advection_term(v::Union{PyObject, Array{Float64, 2}},
    u::Union{PyObject, Array{Float64,1}},m::Int64,n::Int64,h::Float64)

Computes the advection term using upwind schemes
```math
\int_A \mathbf{v} \cdot \nabla u dx 
```
Here $\mathbf{v}$ is a $mn\times 2$ matrix and $u$ is a length $mn$ vector. Zero boundary conditions are assumed. 
$u$ is a vector of length $m\times n$.
"""
function compute_fvm_advection_term(v::Union{PyObject, Array{Float64, 2}},
    u::Union{PyObject, Array{Float64,1}},m::Int64,n::Int64,h::Float64)
    advection_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","advection")
    v,u,m,n,h = convert_to_tensor(Any[v,u,m,n,h], [Float64,Float64,Int64,Int64,Float64])
    advection_(v,u,m,n,h)
end


@doc raw"""
    compute_fvm_advection_matrix(v::Union{PyObject, Array{Float64, 2}},
        bc::Array{Int64, 2},bcval::Union{PyObject, Array{Float64}},m::Int64,n::Int64,h::Float64)

Computes the advection matrix for use in the implicit scheme 
    
```math 
\int_A \mathbf{v} \cdot \nabla u dx 
```
Here `v` is a $2mn$ vector, where the first $mn$ entries corresponds to the first dimension of   $\mathbf{v}$
and the remaining $mn$ entries corresponds to the second dimension. 

It returns a matrix $mn\times mn$ matrix $K$ and an auxilliary term $\mathbf{f}$ due to boundary conditions.

```math 
\int_\Omega \mathbf{v} \cdot \nabla u dx = K \mathbf{u} + \mathbf{f}
```
"""
function compute_fvm_advection_matrix(v::Union{PyObject, Array{Float64, 1}},
    bc::Array{Int64, 2},bcval::Union{PyObject, Array{Float64}},m::Int64,n::Int64,h::Float64)
    @assert length(v)==m*n*2
    @assert length(bcval)==size(bc,1)
    bc = sort(bc, dims = 2)
    implicit_advection_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","implicit_advection", multiple=true)
    uv,bc,bcval,m_,n_,h = convert_to_tensor(Any[v,bc,bcval,m,n,h], [Float64,Int64,Float64,Int64,Int64,Float64])
    ii, jj, vv, rhs = implicit_advection_(uv,bc,bcval,m_,n_,h)
    M = SparseTensor(ii+1, jj+1, vv, m*n, m*n)
    rhs = set_shape(rhs, m*n)
    return M, rhs 
end


@doc raw"""
    compute_fem_source_term1(f::PyObject,
    m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function compute_fem_source_term1(f::PyObject, m::Int64, n::Int64, h::Float64)
    fem_source_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","fem_source")
    f,m_,n_,h = convert_to_tensor(Any[f,m,n,h], [Float64,Int64,Int64,Float64])
    rhs = fem_source_(f,m_,n_,h)
    set_shape(rhs, ((m+1)*(n+1),))
end


@doc raw"""
    compute_fem_source_term(f1::PyObject, f2::PyObject,
    m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function compute_fem_source_term(f1::PyObject, f2::PyObject, m::Int64, n::Int64, h::Float64)
    [compute_fem_source_term1(f1, m, n, h); compute_fem_source_term1(f2, m, n, h)]
end



"""
    eval_grad_on_gauss_pts1(u::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel. 
"""
function eval_grad_on_gauss_pts1(u::PyObject, m::Int64, n::Int64, h::Float64)
    fem_grad_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","fem_grad")
    u,m_,n_,h = convert_to_tensor(Any[u,m,n,h], [Float64,Int64,Int64,Float64])
    out = fem_grad_(u,m_,n_,h)
    return set_shape(out, (4*m*n, 2))
end


"""
    eval_grad_on_gauss_pts(u::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel. 
"""
function eval_grad_on_gauss_pts(u::PyObject, m::Int64, n::Int64, h::Float64)
    r1 = reshape( eval_grad_on_gauss_pts1(u[1:(m+1)*(n+1)], m, n, h), (4*m*n, 1, 2))
    r2 = reshape( eval_grad_on_gauss_pts1(u[(m+1)*(n+1)+1:end], m, n, h), (4*m*n, 1, 2))
    return tf.concat([r1, r2], axis=1)
end



"""
    compute_fem_mass_matrix1(ρ::PyObject,m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function compute_fem_mass_matrix1(ρ::PyObject,m::Int64, n::Int64, h::Float64)
    fem_mass_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","fem_mass", multiple=true)
    rho,m_,n_,h = convert_to_tensor(Any[ρ,m,n,h], [Float64,Int64,Int64,Float64])
    ii, jj, vv = fem_mass_(rho,m_,n_,h)
    SparseTensor(ii+1, jj+1, vv, (m+1)*(n+1), (m+1)*(n+1))
end


@doc raw"""
    compute_fem_mass_matrix(ρ::PyObject,m::Int64, n::Int64, h::Float64)

A differentiable kernel. $\rho$ is a vector of length $4mn$ or $8mn$
"""
function compute_fem_mass_matrix(ρ::PyObject,m::Int64, n::Int64, h::Float64)
    Z = SparseTensor(zeros((m+1)*(n+1), (m+1)*(n+1)))
    if length(ρ)==4*m*n
        M = compute_fem_mass_matrix1(ρ, m, n, h)
        return [M Z 
                Z M]
    elseif length(ρ)==8*m*n
        M1 = compute_fem_mass_matrix1(ρ[1:4*m*n], m, n, h)
        M2 = compute_fem_mass_matrix1(ρ[4*m*n+1:end], m, n, h)
        return [M1 Z 
                Z M2]
    end
end


"""
    compute_fvm_source_term(f::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel.
"""
function compute_fvm_source_term(f::PyObject, m::Int64, n::Int64, h::Float64)
    @assert length(f) == 4*m*n || length(f)==m*n 
    if length(f)==4*m*n 
        f = (f[1:4:end] + f[2:4:end] + f[3:4:end] + f[4:4:end])/4
    end
    f*h^2
end



@doc raw"""
    compute_fem_advection_matrix1(u0::PyObject,v0::PyObject,m::Int64,n::Int64,h::Float64)

Computes the advection term for a scalar function $u$ defined on an FEM grid. The weak form is 

$$\int_\Omega (\mathbf{u}_0 \cdot \nabla u)  \delta u \; d\mathbf{x} = \int_\Omega \left(u_0 \frac{\partial u}{\partial x} \delta u + v_0 \frac{\partial u}{\partial y}  \delta u\right)\; d\mathbf{x}$$

Here $u_0$ and $v_0$ are both vectors of length $4mn$. 

Returns a sparse matrix of size $(m+1)(n+1)\times (m+1)(n+1)$
"""
function compute_fem_advection_matrix1(u0::PyObject,v0::PyObject,m::Int64,n::Int64,h::Float64)
    fem_advection_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","fem_advection", multiple=true)
    u,v,m_,n_,h = convert_to_tensor(Any[u0,v0,m,n,h], [Float64,Float64,Int64,Int64,Float64])
    ii, jj, vv = fem_advection_(u,v,m_,n_,h)
    SparseTensor(ii+1, jj+1, vv, (m+1)*(n+1), (m+1)*(n+1))
end


@doc raw"""
    compute_fem_advection_matrix(u0::PyObject,v0::PyObject,m::Int64,n::Int64,h::Float64)

Computes the advection term for a vector function $\mathbf{u}$

$$\int_\Omega (\mathbf{u}_0 \cdot \nabla \mathbf{u})  \delta \mathbf{u} \; dx =\begin{bmatrix} \int_\Omega u_0 \frac{\partial  u}{\partial x}\delta u\; dx + \int_\Omega v_0 \frac{\partial  u}{\partial y}\delta u\; dx \\ \int_\Omega u_0 \frac{\partial  v}{\partial x}\delta v\; dx + \int_\Omega v_0 \frac{\partial  v}{\partial y}\delta v\; dx  \end{bmatrix}$$

Here $\mathbf{u} = \begin{bmatrix}u \\ v\end{bmatrix}$ is defined on a finite element grid. $u_0$, $v_0$ are vectors of length $4mn$.

Returns a $2(m+1)(n+1)\times 2(m+1)(n+1)$ block diagonal sparse matrix.
"""
function compute_fem_advection_matrix(u0::PyObject,v0::PyObject,m::Int64,n::Int64,h::Float64)
    M = compute_fem_advection_matrix1(u0, v0, m, n, h)
    Z = spzero((m+1)*(n+1))
    [M Z;Z M]
end

@doc raw"""
    compute_fem_laplace_matrix1(K::PyObject, m::Int64, n::Int64, h::Float64)

A differentiable kernel. Only $K\in \mathbb{R}^{4mn}$ is supported.
"""
function compute_fem_laplace_matrix1(kappa::PyObject, m::Int64, n::Int64, h::Float64)
    fem_laplace_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","fem_laplace", multiple=true)
    kappa,m_,n_,h = convert_to_tensor(Any[kappa,m,n,h], [Float64,Int64,Int64,Float64])
    ii, jj, vv = fem_laplace_(kappa,m_,n_,h)
    SparseTensor(ii+1, jj+1, vv, (m+1)*(n+1), (m+1)*(n+1))
end


@doc raw"""
    compute_interaction_term(p::PyObject, m::Int64, n::Int64, h::Float64)

Computes the FVM-FEM interaction term 

```math
 \begin{bmatrix} \int p \frac{\partial \delta u}{\partial x} dx \\  \int p \frac{\partial \delta v}{\partial y}  dy \end{bmatrix} 
```

The input is a vector of length $mn$. The output is a $2(m+1)(n+1)$ vector. 
"""
function compute_interaction_term(p::Union{Array{Float64,1}, PyObject}, m::Int64, n::Int64, h::Float64)
    @assert length(p) == m*n
    interaction_term_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","interaction_term")
    p,m_,n_,h = convert_to_tensor(Any[p,m,n,h], [Float64,Int64,Int64,Float64])
    out = interaction_term_(p,m_,n_,h)
    set_shape(out, (2*(m+1)*(n+1), ))
end