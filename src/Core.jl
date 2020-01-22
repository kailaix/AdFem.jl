export compute_fem_stiffness_matrix,
compute_fem_source_term,
trim_fem,
eval_f_on_gauss_pts,
compute_interaction_matrix,
compute_fvm_source_term,
compute_fvm_mechanics_term,
compute_fluid_tpfa_matrix,
trim_coupled,
compute_elasticity_tangent,
compute_fem_normal_traction_term,
coupled_impose_pressure,
compute_principal_stress_term,
compute_fem_mass_matrix

####################### Mechanics #######################
@doc raw"""
compute_fem_stiffness_matrix(K::Array{Float64,2}, m::Int64, n::Int64, h::Float64)

Computes the term 
```math
\int_{A}\delta \varepsilon :\sigma'\mathrm{d}x = \int_A u_AB^TDB\delta u_A\mathrm{d}x
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
            η = pts[i]; ξ = pts[j]
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
            kk = @. (jj-1)*(m+1) + ii
            kk = [kk; kk .+ (m+1)*(n+1)]
            add(kk, kk, Ω)
        end
    end
    return sparse(I, J, V, (m+1)*(n+1)*2, (m+1)*(n+1)*2)
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
                    
                    k += 1
                    val1 = f1[k] * h^2 * 0.25
                    val2 = f2[k] * h^2 * 0.25
                    
                    rhs[(j-1)*(m+1) + i] += val1 
                    rhs[(j-1)*(m+1) + i+1] += val1 
                    rhs[j*(m+1) + i] += val1 
                    rhs[j*(m+1) + i+1] += val1 
                    
                    rhs[(m+1)*(n+1) + (j-1)*(m+1) + i] += val2
                    rhs[(m+1)*(n+1) + (j-1)*(m+1) + i+1] += val2
                    rhs[(m+1)*(n+1) + j*(m+1) + i] += val2
                    rhs[(m+1)*(n+1) + j*(m+1) + i+1] += val2
                    
                end
            end
            
        end
    end
    return rhs
end

@doc raw"""
trim_fem(A::SparseMatrixCSC{Float64,Int64}, 
bd::Array{Int64}, m::Int64, n::Int64, h::Float64)

Imposes the Dirichlet boundary conditions on the matrix `A`
"""
function trim_fem(A::SparseMatrixCSC{Float64,Int64}, 
    bd::Array{Int64}, m::Int64, n::Int64, h::Float64)
    A[bd,:] = spzeros(length(bd), 2(m+1)*(n+1))
    A[:,bd] = spzeros(2(m+1)*(n+1), length(bd))
    A[bd,bd] = spdiagm(0=>ones(length(bd)))
    A
end

@doc raw"""
eval_f_on_gauss_pts(f::Function, m::Int64, n::Int64, h::Float64)

Evaluates `f` at Gaussian points and return the result as $4mn$ vector `out` (4 Gauss points per element)
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
                    η = pts[p]; ξ = pts[q]
                    x = x1 + ξ*h; y = y1 + η*h
                    k += 1
                    out[k] = f(x, y)
                end
            end
        end
    end
    out
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
compute_fluid_tpfa_matrix(m::Int64, n::Int64, h::Float64)

Computes the term with two-point flux approximation 
```math
\int_{A_i} \Delta p \mathrm{d}x = \sum_{j=1}^{n_{\mathrm{faces}}} (p_j-p_i)
```
![](./assets/tpfa.png)

!!! warning
No flow boundary condition is assumed. 
"""
function compute_fluid_tpfa_matrix(m::Int64, n::Int64, h::Float64)
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
compute_fluid_tpfa_matrix(K::Array{Float64}, m::Int64, n::Int64, h::Float64)

Computes the term with two-point flux approximation with distinct permeability at each cell
```math
\int_{A_i} K_i \Delta p \mathrm{d}x = K_i\sum_{j=1}^{n_{\mathrm{faces}}} (p_j-p_i)
```

"""
function compute_fluid_tpfa_matrix(K::Array{Float64}, m::Int64, n::Int64, h::Float64)
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
        rhs[ii] += t[1]*0.5*h 
        rhs[jj] += t[1]*0.5*h
        rhs[ii+(m+1)*(n+1)] += t[2]*0.5*h
        rhs[jj+(m+1)*(n+1)] += t[2]*0.5*h 
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

`Q` is obtained from [`compute_fluid_tpfa_matrix`](@ref), `M` is obtained from [`compute_fem_stiffness_matrix`](@ref),
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
    compute_principal_stress_term(K::Array{Float64}, u::Array{Float64}, m::Int64, n::Int64, h::Float64)
    
    Compute the principal stress on the Gauss quadrature nodes. 
    """
    function compute_principal_stress_term(K::Array{Float64}, u::Array{Float64}, m::Int64, n::Int64, h::Float64;
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
        for i = 1:m
            for j = 1:n 
                idx = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
                idx = [idx; idx .+ (m+1)*(n+1)]
                uA = u[idx]
                for p = 1:2
                    for q = 1:2
                        Bk = B[(q-1)*2+p,:,:]
                        σ = K * Bk * uA 
                        # σ[1:2] .-= u[(j-1)*m+i+2(m+1)*(n+1)]*b
                        v = eigvals([σ[1] σ[3];σ[3] σ[2]])
                        push!(pval, sqrt(0.5*(v[1]^2+v[2]^2+(v[1]-v[2])^2)))
                    end
                end
            end
        end
        return pval
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
    
    