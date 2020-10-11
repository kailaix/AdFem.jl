export compute_fem_stiffness_matrix,
compute_fem_source_term,
fem_impose_Dirichlet_boundary_condition,
eval_f_on_gauss_pts,
eval_f_on_fvm_pts,
eval_f_on_fem_pts,
compute_fvm_mass_matrix,
compute_interaction_matrix,
compute_fvm_source_term,
compute_fvm_mechanics_term,
compute_fvm_tpfa_matrix,
trim_coupled,
compute_elasticity_tangent,
compute_fem_traction_term,
compute_fem_traction_term1,
compute_fem_normal_traction_term,
coupled_impose_pressure,
compute_von_mises_stress_term,
compute_fem_mass_matrix,
eval_f_on_boundary_node,
eval_f_on_boundary_edge,
compute_fem_mass_matrix1,
compute_fem_stiffness_matrix1,
compute_fem_source_term1,
compute_fem_flux_term1,
fem_impose_Dirichlet_boundary_condition1,
fem_impose_coupled_Dirichlet_boundary_condition,
eval_strain_on_gauss_pts,
eval_stress_on_gauss_pts,
compute_fvm_mass_matrix,
compute_strain_energy_term,
compute_plane_strain_matrix,
compute_fem_laplace_matrix1,
compute_fem_laplace_matrix,
eval_grad_on_gauss_pts1,
eval_grad_on_gauss_pts,
compute_plane_stress_matrix,
eval_f_on_boundary_gauss_pts

####################### Mechanics #######################
@doc raw"""
    compute_fem_stiffness_matrix(K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)

Computes the term 
```math
\int_{A}\delta \varepsilon :\sigma\mathrm{d}x = \int_A u_AB^TKB\delta u_A\mathrm{d}x
```
where the constitutive relation is given by 
```math 
\begin{bmatrix}\sigma_{xx}\\\sigma_{yy}\\\sigma_{xy}\end{bmatrix} = K \begin{bmatrix}\varepsilon_{xx}\\\varepsilon_{yy}\\2\varepsilon_{xy}\end{bmatrix}
```
"""
function compute_fem_stiffness_matrix(K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)
    I = Int64[]; J = Int64[]; V = Float64[]
    function add(ii, jj, kk)
        for  i = 1:length(ii)
            for j = 1:length(jj)
                push!(I,ii[i])
                push!(J,jj[j])
                push!(V,kk[i,j])
            end
        end
    end
    Ω = zeros(8,8)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B = [
            -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
            Ω += B'*K*B*0.25*h^2
        end
    end
    
    for i = 1:m
        for j = 1:n 
            ii = [i;i+1;i;i+1]
            jj = [j;j;j+1;j+1]
            kk = (jj .- 1)*(m+1) + ii
            kk = [kk; kk .+ (m+1)*(n+1)]
            add(kk, kk, Ω)
        end
    end
    return sparse(I, J, V, (m+1)*(n+1)*2, (m+1)*(n+1)*2)
end

@doc raw"""
    compute_fem_stiffness_matrix1(K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)

Computes the term 
```math
\int_{A} (K \nabla u) \cdot \nabla \delta u \mathrm{d}x = \int_A u_A B^T K B \delta u_A\mathrm{d}x
```
Returns a $(m+1)\times (n+1)$ matrix
"""
function compute_fem_stiffness_matrix1(K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)
    I = Int64[]; J = Int64[]; V = Float64[]
    @assert size(K,1)==2 && size(K,2)==2
    function add(ii, jj, kk)
        for  i = 1:length(ii)
            for j = 1:length(jj)
                push!(I,ii[i])
                push!(J,jj[j])
                push!(V,kk[i,j])
            end
        end
    end
    Ω = zeros(4,4)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            ]
            Ω += B'*K*B*0.25*h^2
        end
    end
    
    for i = 1:m
        for j = 1:n 
            ii = [i;i+1;i;i+1]
            jj = [j;j;j+1;j+1]
            kk = (jj .- 1)*(m+1) + ii
            add(kk, kk, Ω)
        end
    end
    return sparse(I, J, V, (m+1)*(n+1), (m+1)*(n+1))
end

@doc raw"""
    compute_fem_source_term(f1::Array{Float64}, f2::Array{Float64},
    m::Int64, n::Int64, h::Float64)

Computes the term 
```math
\int_\Omega \mathbf{f}\cdot\delta u \mathrm{d}x
```
Returns a $2(m+1)(n+1)$ vector. 
"""
function compute_fem_source_term(f1::Array{Float64}, f2::Array{Float64},
    m::Int64, n::Int64, h::Float64)
    rhs = zeros((m+1)*(n+1)*2)
    k = 0
    for i = 1:m
        for j = 1:n
            idx = (j-1)*m + i 
            for p = 1:2
                for q = 1:2
                    ξ = pts[p]; η = pts[q]

                    k = (idx-1)*4 + 2*(q-1) + p
                    val1 = f1[k] * h^2 * 0.25
                    val2 = f2[k] * h^2 * 0.25
                    
                    rhs[(j-1)*(m+1) + i] += val1 * (1-ξ)*(1-η)
                    rhs[(j-1)*(m+1) + i+1] += val1 * ξ*(1-η)
                    rhs[j*(m+1) + i] += val1 * (1-ξ)*η
                    rhs[j*(m+1) + i+1] += val1 * ξ  * η
                    
                    rhs[(m+1)*(n+1) + (j-1)*(m+1) + i] += val2* (1-ξ)*(1-η)
                    rhs[(m+1)*(n+1) + (j-1)*(m+1) + i+1] += val2* ξ*(1-η)
                    rhs[(m+1)*(n+1) + j*(m+1) + i] += val2 * (1-ξ)*η
                    rhs[(m+1)*(n+1) + j*(m+1) + i+1] += val2 * ξ  * η
                    
                end
            end
            
        end
    end
    return rhs
end

@doc raw"""
    compute_fem_source_term1(f::Array{Float64},
    m::Int64, n::Int64, h::Float64)

Computes the term 
```math
\int_\Omega f \delta u dx
```
Returns a $(m+1)\times (n+1)$ vector. `f` is a length $4mn$ vector, given by its values on Gauss points. 
"""
function compute_fem_source_term1(f::Array{Float64}, m::Int64, n::Int64, h::Float64)
    rhs = zeros((m+1)*(n+1))
    for i = 1:m
        for j = 1:n
            idx = (j-1)*m + i 
            for p = 1:2
                for q = 1:2
                    ξ = pts[p]; η = pts[q]
                    k = (idx-1)*4 + 2*(q-1) + p
                    val1 = f[k] * h^2 * 0.25
                    rhs[(j-1)*(m+1) + i] += val1 * (1-ξ)*(1-η)
                    rhs[(j-1)*(m+1) + i+1] += val1 * ξ*(1-η)
                    rhs[j*(m+1) + i] += val1 * (1-ξ)*η
                    rhs[j*(m+1) + i+1] += val1 * ξ  * η
                end
            end
            
        end
    end
    return rhs
end

@doc raw"""
    fem_impose_Dirichlet_boundary_condition(A::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, m::Int64, n::Int64, h::Float64)

Imposes the Dirichlet boundary conditions on the matrix `A`.

Returns 2 matrix, 
```math
\begin{bmatrix}
A_{BB} & A_{BI} \\ 
A_{IB} & A_{II} 
\end{bmatrix} \Rightarrow \begin{bmatrix}
I & 0 \\ 
0 & A_{II} 
\end{bmatrix}, \quad \begin{bmatrix}
0 \\ 
A_{IB} 
\end{bmatrix}
```
"""
function fem_impose_Dirichlet_boundary_condition(A::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    M, N = size(A)
    bd = [bd; bd .+ (m+1)*(n+1)]
    Q = A[:, bd]; Q[bd,:] = spzeros(length(bd), length(bd))
    A[bd,:] = spzeros(length(bd), N) 
    A[:,bd] = spzeros(M, length(bd))
    A[bd,bd] = spdiagm(0=>ones(length(bd)))
    return A, Q  
end

"""
"""
function fem_impose_coupled_Dirichlet_boundary_condition(A::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    bd = [bd; bd .+ (m+1)*(n+1)]
    Q = A[:, bd]; Q[bd,:] = spzeros(length(bd), length(bd))
    A[bd,:] = spzeros(length(bd), 2(m+1)*(n+1)+m*n) 
    A[:,bd] = spzeros(2(m+1)*(n+1)+m*n, length(bd))
    A[bd,bd] = spdiagm(0=>ones(length(bd)))
    return A, Q  
end

@doc raw"""
    fem_impose_Dirichlet_boundary_condition1(A::SparseMatrixCSC{Float64,Int64}, 
        bd::Array{Int64}, m::Int64, n::Int64, h::Float64)

Imposes the Dirichlet boundary conditions on the matrix `A`
Returns 2 matrix, 
```math
\begin{bmatrix}
A_{BB} & A_{BI} \\ 
A_{IB} & A_{II} 
\end{bmatrix} \Rightarrow \begin{bmatrix}
I & 0 \\ 
0 & A_{II} 
\end{bmatrix}, \quad \begin{bmatrix}
0 \\ 
A_{IB} 
\end{bmatrix}
```

`bd` must NOT have duplicates. 
"""
function fem_impose_Dirichlet_boundary_condition1(A::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    M, N = size(A)
    Q = A[:, bd]; Q[bd,:] = spzeros(length(bd), length(bd))
    A[bd,:] = spzeros(length(bd), N)
    A[:,bd] = spzeros(M, length(bd))
    A[bd,bd] = spdiagm(0=>ones(length(bd)))
    return dropzeros(A),  dropzeros(Q)
end

####################### Interaction #######################
@doc raw"""
    compute_interaction_matrix(m::Int64, n::Int64, h::Float64)

Computes the interaction term 
```math
\int_A p \delta \varepsilon_v\mathrm{d}x = \int_A p [1,1,0]B^T\delta u_A\mathrm{d}x
```
Here $\varepsilon_v = \text{tr}\; \varepsilon = \text{div}\; \mathbf{u}$.

The output is a $mn \times 2(m+1)(n+1)$ matrix. 
"""
function compute_interaction_matrix(m::Int64, n::Int64, h::Float64)
    I = Int64[]
    J = Int64[]
    V = Float64[]
    function add(k1, k2, v)
        push!(I, k1)
        push!(J, k2)
        push!(V, v)
    end
    B = zeros(4, 3, 8)
    for i = 1:2
        for j = 1:2
            η = pts[i]; ξ = pts[j]
            B[(j-1)*2+i,:,:] = [
            -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
        end
    end
    for i = 1:m 
        for j = 1:n 
            for p = 1:2
                for q = 1:2
                    idx = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
                    idx = [idx; idx .+ (m+1)*(n+1)]
                    Bk = B[(q-1)*2+p,:,:]
                    Bk = [1. 1. 0.]*Bk # 1 x 8
                    for s = 1:8
                        add((j-1)*m+i, idx[s], Bk[1,s]*0.25*h^2)
                    end
                end
            end
        end
    end
    sparse(I, J, V, m*n, 2(m+1)*(n+1))
end



####################### Fluids #######################
@doc raw"""
    compute_fvm_source_term(f::Array{Float64}, m::Int64, n::Int64, h::Float64)

Computes the source term 
```math
\int_{A_i} f\mathrm{d}x
```
Here $f$ has length $4mn$ or $mn$. In the first case, an average value of four quadrature nodal values of $f$ is used per cell.
"""
function compute_fvm_source_term(f::Array{Float64}, m::Int64, n::Int64, h::Float64)
    out = zeros(m*n)
    @assert length(f) == 4*m*n || length(f)==m*n 
    if length(f)==4*m*n 
        f = (f[1:4:end] + f[2:4:end] + f[3:4:end] + f[4:4:end])/4
    end
    for i = 1:m 
        for j = 1:n 
            k = (j-1)*m + i 
            out[k] = h^2*f[k]
        end
    end
    out
end

@doc raw"""
    compute_fvm_mechanics_term(u::Array{Float64}, m::Int64, n::Int64, h::Float64)

Computes the mechanic interaction term 
```math
\int_{A_i} \varepsilon_v\mathrm{d}x
```
Here 
```math
\varepsilon_v = \mathrm{tr} \varepsilon = \varepsilon_{xx} + \varepsilon_{yy}
```
Numerically, we have 
```math
\varepsilon_v = [1 \ 1 \ 0] B^T \delta u_A
```
"""
function compute_fvm_mechanics_term(u::Array{Float64}, m::Int64, n::Int64, h::Float64)
    out = zeros(m*n)
    B = zeros(4, 3, 8)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B[(j-1)*2+i,:,:] = [
            -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
        end
    end
    for i = 1:m 
        for j = 1:n
            idx = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
            idx = [idx; idx .+ (m+1)*(n+1)]
            for p = 1:2
                for q = 1:2        
                    Bk = B[(q-1)*2+p,:,:]
                    Bk = [1. 1. 0.]*Bk*u[idx]*0.25*h^2
                    out[(j-1)*m+i] += Bk[1,1]
                end
            end
        end
    end
    out
end

"""
    compute_fvm_mechanics_term(u::PyObject, m::Int64, n::Int64, h::Float64)
"""
function compute_fvm_mechanics_term(u::PyObject, m::Int64, n::Int64, h::Float64)
    volumetric_strain_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","volumetric_strain")
    u,m_,n_,h = convert_to_tensor([u,m,n,h], [Float64,Int32,Int32,Float64])
    strain = volumetric_strain_(u,m_,n_,h)
    set_shape(strain, (m*n,))
end

@doc raw"""
    compute_fvm_tpfa_matrix(m::Int64, n::Int64, h::Float64)

Computes the term with two-point flux approximation 
```math
\int_{A_i} \Delta p \mathrm{d}x = \sum_{j=1}^{n_{\mathrm{faces}}} (p_j-p_i)
```
![](./assets/tpfa.png)

!!! warning
    No flow boundary condition is assumed. 
"""
function compute_fvm_tpfa_matrix(m::Int64, n::Int64, h::Float64)
    I = Int64[]; J = Int64[]; V = Float64[]
    function add(i, j, v)
        push!(I, i)
        push!(J, j)
        push!(V, v)
    end
    for i = 1:m 
        for j = 1:n 
            k = (j-1)*m + i
            for (ii,jj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
                if 1<=ii<=m && 1<=jj<=n
                    kp = (jj-1)*m + ii
                    add(k, kp, 1.)
                    add(k, k, -1.)
                end
            end
        end
    end
    sparse(I, J, V, m*n, m*n)
end


@doc raw"""
    compute_fvm_tpfa_matrix(K::Array{Float64}, m::Int64, n::Int64, h::Float64)

Computes the term with two-point flux approximation with distinct permeability at each cell
```math
\int_{A_i} K_i \Delta p \mathrm{d}x = K_i\sum_{j=1}^{n_{\mathrm{faces}}} (p_j-p_i)
```

Note that $K$ is a length $mn$ vector, representing values per cell.
"""
function compute_fvm_tpfa_matrix(K::Array{Float64}, m::Int64, n::Int64, h::Float64)
    I = Int64[]; J = Int64[]; V = Float64[]
    function add(i, j, v)
        push!(I, i)
        push!(J, j)
        push!(V, v)
    end
    for i = 1:m 
        for j = 1:n 
            k = (j-1)*m + i
            for (ii,jj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
                if 1<=ii<=m && 1<=jj<=n
                    kp = (jj-1)*m + ii
                    add(k, kp, K[k])
                    add(k, k, -K[k])
                end
            end
        end
    end
    sparse(I, J, V, m*n, m*n)
end

@doc raw"""
    compute_fvm_tpfa_matrix(K::Array{Float64}, bc::Array{Int64,2}, pval::Array{Float64,1}, m::Int64, n::Int64, h::Float64)

Computes the term with two-point flux approximation with distinct permeability at each cell

```math
\int_{A_i} K_i \Delta p \mathrm{d}x = K_i\sum_{j=1}^{n_{\mathrm{faces}}} (p_j-p_i)
```

Here $K$ is a length $mn$ vector, representing values per cell.

Additionally, Dirichlet boundary conditions are imposed on the boundary edges `bc` (a $N\times 2$ integer matrix), 
i.e., the $i$-th edge has value `pval`. The ghost node method is used for imposing the Dirichlet boundary condition. 
The other boundaries are no-blow boundaries, i.e., $\frac{\partial T}{\partial n} = 0$. 
The function outputs a length $mn$ vector and $mn\times mn$ matrix $M$. 

$$\int_{A_i} K_i \Delta p \mathrm{d}x = f_i + M_{i,:}\mathbf{p}$$

Returns both the sparse matrix `A` and the right hand side `rhs`.

!!! info 
    `K` can also be missing, in which case `K` is treated as a all-one vector. 
"""
function compute_fvm_tpfa_matrix(K::Union{Array{Float64}, Missing}, bc::Array{Int64,2}, pval::Array{Float64,1}, m::Int64, n::Int64, h::Float64)
    bval = Dict{Tuple{Int64, Int64}, Float64}()
    for i = 1:size(bc, 1)
        bval[(minimum(bc[i,:]), maximum(bc[i,:]))] = pval[i] 
    end
    I = Int64[]; J = Int64[]; V = Float64[]
    function add(i, j, v)
        push!(I, i)
        push!(J, j)
        push!(V, v)
    end
    K = coalesce(K, ones(m*n))
    @assert length(K) == m*n # in case users input a 4mn vector. 
    rhs = zeros(m*n)
    for i = 1:m 
        for j = 1:n 
            k = (j-1)*m + i
            for (ii,jj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
                if 1<=ii<=m && 1<=jj<=n
                    kp = (jj-1)*m + ii
                    add(k, kp, K[k])
                    add(k, k, -K[k])
                else  # boundary edges 
                    ed = (0, 0)
                    if ii<=0 # left 
                        ed = ((j-1)*(m+1)+1, j*(m+1)+1)
                    elseif ii>m # right 
                        ed = ((j-1)*(m+1)+m+1, j*(m+1)+m+1)
                    elseif jj<=0 # upper
                        ed = (i, i+1)
                    elseif jj>n # lower
                        ed = (n*(m+1)+i, n*(m+1)+i+1)
                    end
                    if haskey(bval, ed)
                        add(k, k, -2*K[k])
                        rhs[k] += 2*K[k]*bval[ed]
                    end
                end
            end
        end
    end
    sparse(I, J, V, m*n, m*n), rhs
end

@doc raw"""
    compute_fem_traction_term(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)

Computes the traction term 
```math
\int_{\Gamma} t(\mathbf{n})\cdot\delta u \mathrm{d}
```

The number of rows of `t` is equal to the number of edges in `bdedge`. 
The first component of `t` describes the $x$ direction traction, while the second 
component of `t` describes the $y$ direction traction. 

Also see [`compute_fem_normal_traction_term`](@ref). 

![](./assets/traction.png)
"""
function compute_fem_traction_term(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)
    @assert size(t,1)==size(bdedge,1) || size(t,2)==2
    rhs = zeros(2*(m+1)*(n+1))
    for k = 1:size(bdedge, 1)
        ii, jj = bdedge[k,:]
        rhs[ii] += t[k,1]*0.5*h 
        rhs[jj] += t[k,1]*0.5*h
        rhs[ii+(m+1)*(n+1)] += t[k,2]*0.5*h
        rhs[jj+(m+1)*(n+1)] += t[k,2]*0.5*h 
    end
    rhs
end



@doc raw"""
    compute_fem_traction_term1(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)

Computes the traction term 
```math
\int_{\Gamma} t(n) \delta u \mathrm{d}
```

The number of rows of `t` is equal to the number of edges in `bdedge`. 
The output is a length $(m+1)*(n+1)$ vector. 

Also see [`compute_fem_traction_term`](@ref). 
"""
function compute_fem_traction_term1(t::Array{Float64, 1},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)
    @assert length(t)==size(bdedge,1)
    rhs = zeros((m+1)*(n+1))
    for k = 1:size(bdedge, 1)
        ii, jj = bdedge[k,:]
        rhs[ii] += t[k,1]*0.5*h 
        rhs[jj] += t[k,1]*0.5*h
    end
    rhs
end


@doc raw"""
    compute_fem_flux_term1(t::Array{Float64},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)

Computes the traction term 
```math
\int_{\Gamma} q \delta u \mathrm{d}
```
"""
function compute_fem_flux_term1(q::Array{Float64},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)
    @assert length(size(q))==1
    rhs = zeros((m+1)*(n+1))
    for k = 1:size(bdedge, 1)
        ii, jj = bdedge[k,:]
        rhs[ii] += q[k]*0.5*h 
        rhs[jj] += q[k]*0.5*h
    end
    rhs
end

@doc raw"""
    compute_fem_normal_traction_term(t::Array{Float64,1}, bdedge::Array{Int64},
    m::Int64, n::Int64, h::Float64)
    compute_fem_normal_traction_term(t::Float64, bdedge::Array{Int64},
    m::Int64, n::Int64, h::Float64)

Computes the normal traction term 
```math
\int_{\Gamma} t(\mathbf{n})\cdot\delta u \mathrm{d}
```
Here $t(\mathbf{n})\parallel\mathbf{n}$ points **outward** to the domain and the magnitude is given by `t`. 
`bdedge` is a $N\times2$ matrix and each row denotes the indices of two endpoints of the boundary edge. 

See [`compute_fem_traction_term`](@ref) for graphical illustration.
"""
function compute_fem_normal_traction_term(t::Array{Float64,1}, bdedge::Array{Int64},
    m::Int64, n::Int64, h::Float64)
    @assert size(t,1)==size(bdedge,1)
    rhs = zeros(2*(m+1)*(n+1))
    for k = 1:size(bdedge,1)
        normal_ = get_edge_normal(bdedge[k,:], m, n, h)
        ii, jj = bdedge[k,:]
        rhs[ii] += t[k]*normal_[1]*0.5*h 
        rhs[jj] += t[k]*normal_[1]*0.5*h
        rhs[ii+(m+1)*(n+1)] += t[k]*normal_[2]*0.5*h
        rhs[jj+(m+1)*(n+1)] += t[k]*normal_[2]*0.5*h 
    end
    return rhs 
end

function compute_fem_normal_traction_term(t::Float64, bdedge::Array{Int64},
    m::Int64, n::Int64, h::Float64)
    t = t * ones(size(bdedge,1))
    return compute_fem_normal_traction_term(t, bdedge, m, n, h)
end


@doc raw"""
    trim_coupled(pd::PoreData, Q::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, 
    M::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, Δt::Float64, m::Int64, n::Int64, h::Float64)

Assembles matrices from mechanics and flow and assemble the coupled matrix 


$$\begin{bmatrix}
\hat M & -\hat L^T\\
\hat L & \hat Q
\end{bmatrix}$$

`Q` is obtained from [`compute_fvm_tpfa_matrix`](@ref), `M` is obtained from [`compute_fem_stiffness_matrix`](@ref),
and `L` is obtained from [`compute_interaction_matrix`](@ref).
"""
function trim_coupled(pd::PoreData, Q::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, 
    M::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, Δt::Float64, m::Int64, n::Int64, h::Float64)
    
    A22 = 1/pd.M/Δt*h^2*spdiagm(0=>ones(m*n)) -
    pd.kp/pd.Bf/pd.μ*Q
    A21 = pd.b/Δt * L 
    A12 = -A21'
    A11 = M
    
    A = [A11 A12
    A21 A22]
    A[bd,:] = spzeros(length(bd), 2(m+1)*(n+1)+m*n)
    A[:, bd] = spzeros(2(m+1)*(n+1)+m*n, length(bd))
    A[bd, bd] = spdiagm(0=>ones(length(bd)))
    dropzeros(A)
end

"""
    coupled_impose_pressure(A::SparseMatrixCSC{Float64,Int64}, pnode::Array{Int64}, 
    m::Int64, n::Int64, h::Float64)

Returns a trimmed matrix.
"""
function coupled_impose_pressure(A::SparseMatrixCSC{Float64,Int64}, pnode::Array{Int64}, 
    m::Int64, n::Int64, h::Float64)
    pnode = pnode.+2(m+1)*(n+1)
    A[pnode, :] = spzeros(length(pnode), 2(m+1)*(n+1)+m*n)
    A[pnode, pnode] = spdiagm(0=>ones(length(pnode)))
    A
end

function get_gauss_points(m, n, h)
    pts_ = []
    for i = 1:m 
        for j = 1:n 
            for p = 1:2
                for q = 1:2
                    x = pts[p]*h+(i-1)*h
                    y = pts[q]*h+(j-1)*h
                    push!(pts_, [x y])
                end
            end
        end
    end
    vcat(pts_...)
end

@doc raw"""
    compute_plane_strain_matrix(E::Float64, ν::Float64)

Computes the stiffness matrix for 2D plane strain. The matrix is given by 

$$\frac{E(1-\nu)}{(1+\nu)(1-2\nu)}\begin{bmatrix}
1 & \frac{\nu}{1-\nu} & \frac{\nu}{1-\nu}\\ 
\frac{\nu}{1-\nu} & 1 & \frac{\nu}{1-\nu} \\ 
\frac{\nu}{1-\nu} & \frac{\nu}{1-\nu} & 1
\end{bmatrix}$$
"""
function compute_plane_strain_matrix(E::Float64, ν::Float64)
    E*(1-ν)/(1+ν)/(1-2ν)*[
    1 ν/(1-ν) ν/(1-ν)
    ν/(1-ν) 1 ν/(1-ν)
    ν/(1-ν) ν/(1-ν) 1
    ] 
end


@doc raw"""
    compute_plane_stress_matrix(E::Float64, ν::Float64)

Computes the stiffness matrix for 2D plane stress. The matrix is given by 

$$\frac{E}{(1+\nu)(1-2\nu)}\begin{bmatrix}
1-\nu & \nu & 0\\ 
\nu & 1 & 0 \\ 
0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix}$$
"""
function compute_plane_stress_matrix(E::Float64, ν::Float64)
    E/(1+ν)/(1-2ν)*[
        1-ν ν 0.0
        ν 1-ν 0.0
        0.0 0.0 (1-2ν)/2
    ]
end


"""
    compute_plane_strain_matrix(E::Union{PyObject, Array{Float64, 1}}, nu::Union{PyObject, Array{Float64, 1}})
"""
function compute_plane_strain_matrix(E::Union{PyObject, Array{Float64, 1}}, nu::Union{PyObject, Array{Float64, 1}})
    mode = 0
    N = length(nu)
    @assert length(E)==N
    plane_strain_and_stress_ = load_op_and_grad(PoreFlow.libmfem,"plane_strain_and_stress")
    e,nu,mode = convert_to_tensor(Any[E,nu,mode], [Float64,Float64,Int32])
    out = plane_strain_and_stress_(e,nu,mode)
    set_shape(out, (N, 3, 3))
end

"""
    compute_plane_stress_matrix(E::Union{PyObject, Array{Float64, 1}}, nu::Union{PyObject, Array{Float64, 1}})
"""
function compute_plane_stress_matrix(E::Union{PyObject, Array{Float64, 1}}, nu::Union{PyObject, Array{Float64, 1}})
    mode = 1
    N = length(nu)
    @assert length(E)==N
    plane_strain_and_stress_ = load_op_and_grad(PoreFlow.libmfem,"plane_strain_and_stress")
    e,nu,mode = convert_to_tensor(Any[E,nu,mode], [Float64,Float64,Int32])
    out = plane_strain_and_stress_(e,nu,mode)
    set_shape(out, (N, 3, 3))
end



"""
    compute_von_mises_stress_term(K::Array{Float64}, u::Array{Float64}, m::Int64, n::Int64, h::Float64)

Compute the [von Mises stress](https://en.wikipedia.org/wiki/Von_Mises_yield_criterion#Multi-axial_(2D_or_3D)_stress) on the Gauss quadrature nodes. 
"""
function compute_von_mises_stress_term(K::Array{Float64}, u::Array{Float64}, m::Int64, n::Int64, h::Float64;
    b::Float64 = 1.0)
    I = Int64[]; J = Int64[]; V = Float64[]
    B = zeros(4, 3, 8)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B[(j-1)*2+i,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
                0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
        end
    end
    
    pval = Float64[]
    for j = 1:n 
        for i = 1:m
            idx = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
            idx = [idx; idx .+ (m+1)*(n+1)]
            uA = u[idx]
            for p = 1:2
                for q = 1:2
                    Bk = B[(q-1)*2+p,:,:]
                    σ = K * Bk * uA # 3 components
                    σ11, σ22, σ12 = σ
                    σv = sqrt(σ11^2 - σ11*σ22 + σ22^2 + 3σ12^2)
                    push!(pval, σv)
                end
            end
        end
    end
    return pval
end

@doc raw"""
    compute_von_mises_stress_term(Se::Array{Float64,2},  m::Int64, n::Int64, h::Float64)

`Se` is a $4mn\times3$ array that stores the stress data at each Gauss point. 
"""
function compute_von_mises_stress_term(Se::Array{Float64,2},  m::Int64, n::Int64, h::Float64)
    pval = Float64[]
    S = zeros(4*m*n)
    for i = 1:4*m*n 
        σ11, σ22, σ12 = Se[i,:]
        S[i] = sqrt(σ11^2 - σ11*σ22 + σ22^2 + 3σ12^2)
    end
    return (S[1:4:end]+S[2:4:end]+S[3:4:end]+S[4:4:end])/4
end

@doc raw"""
    compute_fem_mass_matrix1(ρ::Array{Float64,1}, m::Int64, n::Int64, h::Float64)

Computes the mass matrix for a scalar value $u$
```math
\int_A \rho u \delta u \mathrm{d} x
```
The output is a $(m+1)(n+1)\times (m+1)(n+1)$ sparse matrix. 
"""
function compute_fem_mass_matrix1(ρ::Array{Float64}, m::Int64, n::Int64, h::Float64)
    I = Int64[]; J = Int64[]; V = Float64[]
    function add(ii, jj, vv)
        for i = 1:4
            for j = 1:4
                push!(I,ii[i])
                push!(J,jj[j])
                push!(V,vv[i,j])
            end
        end
    end
    
    Me = zeros(4, 4, 4)
    for p = 1:2
        for q = 1:2
            ξ = pts[p]; η = pts[q]
            A = zeros(4)
            A[1] = (1-ξ)*(1-η)
            A[2] = ξ*(1-η)
            A[3] = (1-ξ)*η
            A[4] = ξ*η
            Me[(p-1)*2+q,:,:] = A*A'*0.25*h^2
        end
    end
    for i = 1:m
        for j = 1:n 
            elem_idx = (j-1)*m + i 
            idx = [i+(j-1)*(m+1); i+1+(j-1)*(m+1); i+j*(m+1); i+1+j*(m+1)]
            for p = 1:2
                for q = 1:2
                    k = (elem_idx-1)*4 + 2*(q-1) + p
                    ρ_ = ρ[k]
                    add(idx, idx, ρ_*Me[(p-1)*2+q,:,:])
                end
            end
        end
    end
    sparse(I, J, V, (m+1)*(n+1), (m+1)*(n+1))
end

@doc raw"""
    compute_fem_mass_matrix1(m::Int64, n::Int64, h::Float64)

Computes the mass matrix for a scalar value $u$
```math
\int_A u \delta u \mathrm{d} x
```
The output is a $(m+1)(n+1)\times (m+1)(n+1)$ sparse matrix. 
"""
function compute_fem_mass_matrix1(m::Int64, n::Int64, h::Float64)
    ρ = ones(4*m*n)
    compute_fem_mass_matrix1(ρ, m, n, h)
end

@doc raw"""
    compute_fem_mass_matrix(m::Int64, n::Int64, h::Float64)

Computes the finite element mass matrix 

```math
\int_{\Omega} u \delta u \mathrm{d}x
```

The matrix size is $2(m+1)(n+1) \times 2(m+1)(n+1)$.
"""
function compute_fem_mass_matrix(m::Int64, n::Int64, h::Float64)
    M = compute_fem_mass_matrix1(m, n, h)    
    Z = spzeros((m+1)*(n+1), (m+1)*(n+1))
    [M Z;Z M]
end
    
    
@doc raw"""
    eval_f_on_gauss_pts(f::Function, m::Int64, n::Int64, h::Float64; tensor_input::Bool = false)

Evaluates `f` at Gaussian points and return the result as $4mn$ vector `out` (4 Gauss points per element)

If `tensor_input = true`, the function `f` is assumed to map a tensor to a tensor output.

![](./assets/gauss.png)
"""
function eval_f_on_gauss_pts(f::Function, m::Int64, n::Int64, h::Float64; tensor_input::Bool = false)
    xy = gauss_nodes(m, n, h)
    x, y = xy[:,1], xy[:,2]
    if tensor_input
        return f(constant(x), constant(y))
    end
    return f.(x, y)
end

@doc raw"""
    eval_f_on_fem_pts(f::Function, m::Int64, n::Int64, h::Float64; tensor_input::Bool = false)

Returns $f(x_i, y_i)$ where $(x_i,y_i)$ are FEM nodes. 

If `tensor_input = true`, the function `f` is assumed to map a tensor to a tensor output.
"""
function eval_f_on_fem_pts(f::Function, m::Int64, n::Int64, h::Float64; tensor_input::Bool = false)
    xy = fem_nodes(m, n, h)
    x, y = xy[:,1], xy[:,2]
    if tensor_input
        return f(constant(x), constant(y))
    end
    return f.(x, y)
end

@doc raw"""
    eval_f_on_fvm_pts(f::Function, m::Int64, n::Int64, h::Float64; tensor_input::Bool = false)

Returns $f(x_i, y_i)$ where $(x_i,y_i)$ are FVM nodes. 

If `tensor_input = true`, the function `f` is assumed to map a tensor to a tensor output.
"""
function eval_f_on_fvm_pts(f::Function, m::Int64, n::Int64, h::Float64; tensor_input::Bool = false)
    xy = fvm_nodes(m, n, h)
    x, y = xy[:,1], xy[:,2]
    if tensor_input
        return f(constant(x), constant(y))
    end
    return f.(x, y)
end


@doc raw"""
    eval_f_on_boundary_node(f::Function, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)

Returns a vector of the same length as `bdnode` whose entries corresponding to `bdnode` nodes
are filled with values computed from `f`.

`f` has the following signature 
```
f(x::Float64, y::Float64)::Float64
```
"""
function eval_f_on_boundary_node(f::Function, bdnode::Array{Int64}, m::Int64, n::Int64, h::Float64)
    out = zeros(length(bdnode))
    for i = 1:length(bdnode)
        i1, j1 = femidx(bdnode[i], m)
        out[i] = f((i1-1)*h, (j1-1)*h)
    end
    out 
end

@doc raw"""
    eval_f_on_boundary_edge(f::Function, bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)

Returns a vector of the same length as `bdedge` whose entries corresponding to `bdedge` nodes
are filled with values computed from `f`.

`f` has the following signature 
```
f(x::Float64, y::Float64)::Float64
```
"""
function eval_f_on_boundary_edge(f::Function, bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)
    out = zeros(size(bdedge,1))
    for i = 1:size(bdedge,1)
        i1, j1 = femidx(bdedge[i,1], m)
        i2, j2 = femidx(bdedge[i,2], m)
        x = ((i1-1)*h + (i2-1)*h)/2
        y = ((j1-1)*h + (j2-1)*h)/2
        out[i] = f(x, y)
    end
    out 
end

@doc raw"""
    eval_strain_on_gauss_pts(u::Array{Float64}, m::Int64, n::Int64, h::Float64)

Computes the strain on Gauss points. 
Returns a $4mn\times3$ matrix, where each row denotes $(\varepsilon_{11}, \varepsilon_{22}, 2\varepsilon_{12})$
at the corresponding Gauss point. 
"""
function eval_strain_on_gauss_pts(u::Array{Float64}, m::Int64, n::Int64, h::Float64)
    I = Int64[]
    J = Int64[]
    V = Float64[]
    function add(k1, k2, v)
        push!(I, k1)
        push!(J, k2)
        push!(V, v)
    end
    B = zeros(4, 3, 8)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B[(i-1)*2+j,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
                0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
        end
    end
    strain = zeros(4m*n, 3)
    for i = 1:m 
        for j = 1:n 
            for p = 1:2
                for q = 1:2
                    idx = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
                    idx = [idx; idx .+ (m+1)*(n+1)]
                    Bk = B[(p-1)*2+q,:,:] # 3 x 8
                    strain[4*((j-1)*m+i-1)+(p-1)*2+q,:] = Bk * u[idx]
                end
            end
        end
    end
    strain
end    




@doc raw"""
    eval_stress_on_gauss_pts(u::Array{Float64}, K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)

Returns the stress on the Gauss points for elasticity. 
"""
function eval_stress_on_gauss_pts(u::Array{Float64}, K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)
    strain = eval_strain_on_gauss_pts(u, m, n, h)
    strain * K 
end

@doc raw"""
    compute_fvm_mass_matrix(m::Int64, n::Int64, h::Float64)

Returns the FVM mass matrix 
```math
\int_{A_i} p_i \mathrm{d}x = h^2 p_i 
```
"""
function compute_fvm_mass_matrix(m::Int64, n::Int64, h::Float64)
    return spdiagm(0=>ones(m*n))*h^2
end


@doc raw"""
    compute_strain_energy_term(S::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)

Computes the strain energy 
```math
\int_{A} \sigma : \delta \varepsilon \mathrm{d}x
```
where $\sigma$ is provided by `S`, a $4mn \times 3$ matrix. 
The values $\sigma_{11}, \sigma_{22}, \sigma_{12}$ are defined on 4 Gauss points per element. 
"""
function compute_strain_energy_term(S::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)
    f = zeros(2(m+1)*(n+1))
    
    B = zeros(4, 3, 8)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B[(i-1)*2+j,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
                0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
        end
    end

    for i = 1:m
        for j = 1:n 
            elem = (j-1)*m + i 
            σ = S[4(elem-1)+1:4elem, :] # 4×3
            dof = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
            dof = [dof; dof .+ (m+1)*(n+1)]
            for p = 1:2
                for q = 1:2
                    idx = 2(q-1) + p
                    f[dof] += (σ[idx,:]' * B[idx,:,:])[:]*0.25*h^2 # length 8 vector
                end
            end
        end
    end
    f
end


@doc raw"""
    compute_strain_energy_term1(S::PyObject, m::Int64, n::Int64, h::Float64)

Computes the strain energy 
```math
\int_{A} \sigma : \delta \varepsilon \mathrm{d}x
```
where $\sigma$ is provided by `S`, a $4mn \times 2$ matrix. 
The values $\sigma_{31}, \sigma_{32}$ are defined on 4 Gauss points per element. 
"""
function compute_strain_energy_term1(S::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)
    @assert size(S,2)==2
    f = zeros((m+1)*(n+1))

    B = zeros(4, 2, 4)
    for p = 1:2
        for q = 1:2
            ξ = pts[p]; η = pts[q]
            B[(q-1)*2+p,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            ]
        end
    end

    for i = 1:m
        for j = 1:n 
            elem = (j-1)*m + i 
            σ = S[4(elem-1)+1:4elem, :] # 4×2
            dof = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
            for p = 1:2
                for q = 1:2
                    idx = 2(q-1) + p
                    f[dof] += (σ[idx,:]' * B[idx,:,:])[:]*0.25*h^2 # length 8 vector
                end
            end
        end
    end
    f
end


@doc raw"""
    compute_fem_laplace_matrix1(K::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)

Computes the coefficient matrix for 

```math
\int_\Omega K \nabla u \cdot \nabla (\delta u) \; dx 
```
Here $K\in \mathbf{R}^{2\times 2}$, $u$ is a scalar variable, and `K` is a $4mn \times 2 \times 2$ matrix. 

Returns a $(m+1)(n+1)\times (m+1)(n+1)$ sparse matrix. 
"""
function compute_fem_laplace_matrix1(K::Array{Float64, 3}, m::Int64, n::Int64, h::Float64)
    @assert size(K,2)==size(K,3)==2
    
    B = zeros(4, 2, 4)
    for q = 1:2
        for p = 1:2
            ξ = pts[p]; η = pts[q]
            B[(q-1)*2+p,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            ]
        end
    end

    I = Int64[]
    J = Int64[]
    V = Float64[]
    function add(k1, k2, v)
        push!(I, k1)
        push!(J, k2)
        push!(V, v)
    end
    k = 0
    for j = 1:n 
        for i = 1:m 
            for q = 1:2
                for p = 1:2
                    k += 1
                    B0 = B[(q-1)*2+p,:,:]
                    Ω = B0'*K[k,:,:]*B0*0.25*h^2
                    idx = [(j-1)*(m+1)+i; (j-1)*(m+1)+i+1; j*(m+1)+i; j*(m+1)+i+1]
                    for i_ = 1:4
                        for j_ = 1:4
                            add(idx[i_], idx[j_], Ω[i_, j_])
                        end
                    end
                end
            end
        end
    end
    
    return sparse(I,J,V,(m+1)*(n+1),(m+1)*(n+1))
end

"""
    compute_fem_laplace_matrix1(K::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)

`K` is duplicated on each Gauss point. 
"""
function compute_fem_laplace_matrix1(K::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)
    Knew = zeros(4m*n, 2, 2)
    for i = 1:4*m*n 
        Knew[i,:,:] = K 
    end
    return compute_fem_laplace_matrix1(Knew, m, n, h)
end

@doc raw"""
    compute_fem_laplace_matrix1(K::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)

Computes the coefficient matrix for 

```math
\int_\Omega K\nabla u \cdot \nabla (\delta u) \; dx 
``` 

Here `K` is a vector with length $4mn$ (defined on Gauss points). 

Returns a $(m+1)(n+1)\times (m+1)(n+1)$ sparse matrix. 
"""
function compute_fem_laplace_matrix1(K::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)
    @assert length(K)==4*m*n
    Knew = zeros(4m*n, 2, 2)
    for i = 1:4*m*n 
        Knew[i,:,:] = K[i] * diagm(0=>ones(2)) 
    end
    return compute_fem_laplace_matrix1(Knew, m, n, h)
end

@doc raw"""
    compute_fem_laplace_matrix1(m::Int64, n::Int64, h::Float64)

Computes the coefficient matrix for 

```math
\int_\Omega \nabla u \cdot \nabla (\delta u) \; dx 
```

Returns a $(m+1)(n+1)\times (m+1)(n+1)$ sparse matrix. 
"""
function compute_fem_laplace_matrix1(m::Int64, n::Int64, h::Float64)
    Knew = zeros(4m*n, 2, 2)
    for i = 1:4*m*n 
        Knew[i,:,:] = diagm(0=>ones(2)) 
    end
    return compute_fem_laplace_matrix1(Knew, m, n, h)
end


@doc raw"""
    compute_fem_laplace_matrix(m::Int64, n::Int64, h::Float64)

Computes the coefficient matrix for 

```math
\int_\Omega \nabla \mathbf{u} \cdot \nabla (\delta \mathbf{u}) \; dx 
```

Here

$$\mathbf{u}  = \begin{bmatrix} u \\ v \end{bmatrix}$$

and 

$$\nabla \mathbf{u} = \begin{bmatrix}u_x & u_y \\ v_x & v_y \end{bmatrix}$$

Returns a $2(m+1)(n+1)\times 2(m+1)(n+1)$ sparse matrix. 
"""
function compute_fem_laplace_matrix(m::Int64, n::Int64, h::Float64)
    K = compute_fem_laplace_matrix1(m, n, h)
    Z = spzeros((m+1)*(n+1), (m+1)*(n+1))
    [K Z;Z K]
end

@doc raw"""
    compute_fem_laplace_matrix(K::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)

Computes the coefficient matrix for 

```math
\int_\Omega K \nabla \mathbf{u} \cdot \nabla (\delta \mathbf{u}) \; dx 
```

Here $K$ is a scalar defined on Gauss points. `K` is a vector of length $4mn$
"""
function compute_fem_laplace_matrix(K::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)
    K1 = compute_fem_laplace_matrix1(K, m, n, h)
    K2 = compute_fem_laplace_matrix1(K, m, n, h)
    Z = spzeros((m+1)*(n+1), (m+1)*(n+1))
    [K1 Z;Z K2]
end


@doc raw"""
    eval_grad_on_gauss_pts1(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64)


Evaluates $\nabla u$ on each Gauss point. Here $u$ is a scalar function. 

The input `u` is a vector of length $(m+1)*(n+1)$. The output is a matrix of size $4mn\times 2$. 
"""
function eval_grad_on_gauss_pts1(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64)
    @assert length(u) == (m+1)*(n+1)
    ret = zeros(4*m*n, 2)
    B = zeros(4, 2, 4)
    for q = 1:2
        for p = 1:2
            k = (q-1)*2 + p 
            ξ = pts[p]; η = pts[q]
            B[k, :, :] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            ]
        end
    end
    ret_idx = 0
    for j = 1:n 
        for i = 1:m 
            idx = [(j-1)*(m+1)+i; (j-1)*(m+1)+i+1; j*(m+1)+i; j*(m+1)+i+1 ]
            uA = u[idx]
            for q = 1:2
                for p = 1:2
                    k = (q-1)*2 + p 
                    B0 = B[k, :, :]
                    ret_idx += 1
                    ret[ret_idx, :] = B0 * uA
                end
            end
        end
    end
    return ret 
end

@doc raw"""
    eval_grad_on_gauss_pts(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64)


Evaluates $\nabla u$ on each Gauss point. Here $\mathbf{u} = (u, v)$.

$$\texttt{g[i,:,:]} = \begin{bmatrix} u_x & u_y\\ v_x & v_y \end{bmatrix}$$

The input `u` is a vector of length $2(m+1)*(n+1)$. The output is a matrix of size $4mn\times 2 \times 2$. 
"""
function eval_grad_on_gauss_pts(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64)
    r1 = eval_grad_on_gauss_pts1(u[1:(m+1)*(n+1)], m, n, h)
    r2 = eval_grad_on_gauss_pts1(u[(m+1)*(n+1)+1:end], m, n, h)
    ret = zeros(4*m*n, 2, 2)
    for i = 1:4*m*n
        ret[i, 1, :] = r1[i,:]
        ret[i, 2, :] = r2[i,:] 
    end
    return ret 
end

@doc raw"""
    compute_interaction_term(p::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)

Computes the FVM-FEM interaction term 

```math
 \begin{bmatrix} \int p \frac{\partial \delta u}{\partial x} dx \\  \int p \frac{\partial \delta v}{\partial y}  dy \end{bmatrix} 
```

The input is a vector of length $mn$. The output is a $2(m+1)(n+1)$ vector. 
"""
function compute_interaction_term(pres::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)
    @assert length(pres)==m*n 
    rhs = zeros((m+1)*(n+1)*2)
    B = zeros(4, 2, 8)
    for i = 1:2
        for j = 1:2
            η = pts[i]; ξ = pts[j]
            B[(j-1)*2+i,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
                0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            ]
        end
    end
    for i = 1:m 
        for j = 1:n 
            pA = pres[(j-1)*m+i]
            for p = 1:2
                for q = 1:2
                    idx1 = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
                    idx2 = idx1 .+ (m+1)*(n+1)
                    Bk = B[(q-1)*2+p,:,:]
                    rhs[idx1] += B[(q-1)*2+p,1,1:4] * pA * h * h * 0.25
                    rhs[idx2] += B[(q-1)*2+p,2,5:end] * pA * h * h * 0.25
                end
            end
        end
    end
    rhs
end


function compute_fem_advection_matrix1(u0::Array{Float64, 1},v0::Array{Float64, 1},m::Int64,n::Int64,h::Float64)
    @assert length(u0) == 4*m*n
    @assert length(v0) == 4*m*n

    B = zeros(4, 2, 4)
    for q = 1:2
        for p = 1:2
            ξ = pts[p]; η = pts[q]
            B[(q-1)*2+p,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
            ]
        end
    end

    I = Int64[]
    J = Int64[]
    V = Float64[]
    function add(k1, k2, v)
        push!(I, k1)
        push!(J, k2)
        push!(V, v)
    end
    k = 0
    for j = 1:n 
        for i = 1:m 
            for q = 1:2
                for p = 1:2
                    k += 1
                    B0 = B[(q-1)*2+p,:,:]
                    Ω = B0'*K[k,:,:]*B0*0.25*h^2
                    idx = [(j-1)*(m+1)+i; (j-1)*(m+1)+i+1; j*(m+1)+i; j*(m+1)+i+1]
                    for i_ = 1:4
                        for j_ = 1:4
                            add(idx[i_], idx[j_], Ω[i_, j_])
                        end
                    end
                end
            end
        end
    end
end