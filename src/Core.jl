export compute_fem_stiffness_matrix,
compute_fem_source_term,
fem_impose_Dirichlet_boundary_condition,
eval_f_on_gauss_pts,
compute_fvm_mass_matrix,
compute_interaction_matrix,
compute_fvm_source_term,
compute_fvm_mechanics_term,
compute_fvm_tpfa_matrix,
trim_coupled,
compute_elasticity_tangent,
compute_fem_traction_term,
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
compute_strain_energy_term

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
\int_\Omega f\cdot\delta u dx
```
"""
function compute_fem_source_term(f1::Array{Float64}, f2::Array{Float64},
    m::Int64, n::Int64, h::Float64)
    rhs = zeros((m+1)*(n+1)*2)
    k = 0
    for i = 1:m
        for j = 1:n
            for p = 1:2
                for q = 1:2
                    ξ = pts[p]; η = pts[q]

                    k += 1
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
Returns a $(m+1)\times (n+1)$ vector. 
"""
function compute_fem_source_term1(f::Array{Float64}, m::Int64, n::Int64, h::Float64)
    rhs = zeros((m+1)*(n+1))
    k = 0
    for i = 1:m
        for j = 1:n
            for p = 1:2
                for q = 1:2
                    ξ = pts[p]; η = pts[q]
                    k += 1
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
    bd = [bd; bd .+ (m+1)*(n+1)]
    rhs = zeros(2*(m+1)*(n+1))
    Q = A[:, bd]; Q[bd,:] = spzeros(length(bd), length(bd))
    A[bd,:] = spzeros(length(bd), 2(m+1)*(n+1))
    A[:,bd] = spzeros(2(m+1)*(n+1), length(bd))
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
"""
function fem_impose_Dirichlet_boundary_condition1(A::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    rhs = zeros((m+1)*(n+1))
    Q = A[:, bd]; Q[bd,:] = spzeros(length(bd), length(bd))
    A[bd,:] = spzeros(length(bd), (m+1)*(n+1))
    A[:,bd] = spzeros((m+1)*(n+1), length(bd))
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
"""
function compute_fvm_source_term(f::Array{Float64}, m::Int64, n::Int64, h::Float64)
    out = zeros(m*n)
    @assert length(f) == 4*m*n 
    for i = 1:m 
        for j = 1:n 
            k = (j-1)*m + i 
            out[k] = 0.25*h^2*sum(f[4*(k-1)+1:4*k])
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
    compute_fem_traction_term(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)

Computes the traction term 
```math
\int_{\Gamma} t(\mathbf{n})\cdot\delta u \mathrm{d}
```

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

"""
    compute_elasticity_tangent(E::Float64, ν::Float64)

Computes the elasticity matrix for 2D plane strain
"""
function compute_elasticity_tangent(E::Float64, ν::Float64)
    E*(1-ν)/(1+ν)/(1-2ν)*[
    1 ν/(1-ν) ν/(1-ν)
    ν/(1-ν) 1 ν/(1-ν)
    ν/(1-ν) ν/(1-ν) 1
    ]
    # E/(1+ν)/(1-2ν)*[
    #     1-ν ν 0.0
    #     ν 1-ν 0.0
    #     0.0 0.0 (1-2ν)/2
    # ]
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
    compute_fem_mass_matrix1(ρ::Array{Float64}, m::Int64, n::Int64, h::Float64)

Computes the mass matrix for a scalar value $u$
```math
\int_A \rho u \delta u \mathrm{d} x
```
The output is a $(m+1)*(n+1)$ sparse matrix. 
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
    k = 0
    for i = 1:m
        for j = 1:n 
            idx = [i+(j-1)*(m+1); i+1+(j-1)*(m+1); i+j*(m+1); i+1+j*(m+1)]
            for p = 1:2
                for q = 1:2
                    k += 1
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
```
\int_A u \delta u \mathrm{d} x
```
The output is a $(m+1)*(n+1)$ sparse matrix. 
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
    I = Int64[]; J = Int64[]; V = Float64[]
    function add!(i, j)
        idx = [i+(j-1)*(m+1); i+1+(j-1)*(m+1); i+j*(m+1); i+1+j*(m+1)]
        for l1 = 1:4
            for l2 = 1:4
                push!(I, idx[l1]); push!(J, idx[l2]); push!(V, A[l1]*A[l2])
                push!(I, idx[l1]+(m+1)*(n+1)); push!(J, idx[l2]+(m+1)*(n+1)); push!(V, A[l1]*A[l2])
            end
        end
    end
    A = zeros(4)
    for p = 1:2
        for q = 1:2
            ξ = pts[p]; η = pts[q]
            A[1] += (1-ξ)*(1-η)*0.25*h^2
            A[2] += ξ*(1-η)*0.25*h^2
            A[3] += (1-ξ)*η*0.25*h^2
            A[4] += ξ*η*0.25*h^2
        end
    end
    
    for i = 1:m
        for j = 1:n 
            add!(i, j)
        end
    end
    sparse(I, J, V, 2(m+1)*(n+1), 2(m+1)*(n+1))
end
    
    
@doc raw"""
    eval_f_on_gauss_pts(f::Function, m::Int64, n::Int64, h::Float64)

Evaluates `f` at Gaussian points and return the result as $4mn$ vector `out` (4 Gauss points per element)

![](./assets/gauss.png)
"""
function eval_f_on_gauss_pts(f::Function, m::Int64, n::Int64, h::Float64)
    out = zeros(4*m*n)
    k = 0
    for i = 1:m 
        for j = 1:n 
            x1 = (i-1)*h 
            y1 = (j-1)*h
            for p = 1:2
                for q = 1:2
                    ξ = pts[p]; η = pts[q]
                    x = x1 + ξ*h; y = y1 + η*h
                    k += 1
                    out[k] = f(x, y)
                end
            end
        end
    end
    out
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
Returns a $4mn \times 3$ matrix, where each row denotes $(\varepsilon_{11}, \varepsilon_{22}, 2\varepsilon_{12})$
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