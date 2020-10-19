using Revise
using AdFem
using PyCall
using LinearAlgebra
np = pyimport("numpy")

# Domain information 
NT = 50
Δt = 1/NT
n = 50
m = n 
h = 1.0/n 
bdnode = Int64[]
for i = 1:m+1
    for j = 1:n+1
        if i==1 || i==m+1 || j==1|| j==n+1
            push!(bdnode, (j-1)*(m+1)+i)
        end
    end
end

# Physical parameters
b = 1.0
H = [1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 0.5]
Q = compute_fvm_tpfa_matrix(m, n, h)
K = compute_fem_stiffness_matrix(H, m, n, h)
L = compute_interaction_matrix(m, n, h)
M = compute_fvm_mass_matrix(m, n, h)
A = [K -b*L'
b*L/Δt 1/Δt*M-Q]
A, Abd = fem_impose_coupled_Dirichlet_boundary_condition(A, bdnode, m, n, h)

U = zeros(m*n+2(m+1)*(n+1), NT+1)
x = Float64[]; y = Float64[]
for j = 1:n+1
    for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
    end
end
    
injection = (div(n,2)-1)*m + 3
production = (div(n,2)-1)*m + m-3


K, σY = 0.5, 1.0
σ0 = zeros(4*m*n, 3)
α0 = zeros(4*m*n)
for i = 1:NT 
    t = i*Δt
    @info i
        
    bdval = zeros(2*length(bdnode))
    up = copy(U[:, i])
    ε0 = eval_strain_on_gauss_pts(U[1:2(m+1)*(n+1),i], m, n, h)
    iter = 0
    while true
        iter += 1
        # @info size(ε0), size(σ0), size(α0)
        global fint, stiff, α, σ = compute_planestressplasticity_stress_and_stiffness_matrix(
            up[1:2(m+1)*(n+1)], ε0, σ0, α0, K, σY, H, m, n, h
        )
        
        rhs1 = fint - b*L'*up[2(m+1)*(n+1)+1:end]
        rhs2 =  b*L/Δt*(up[1:2(m+1)*(n+1)] - U[1:2(m+1)*(n+1), i]) + 
                M * (up[2(m+1)*(n+1)+1:end] - U[2(m+1)*(n+1)+1:end,i])/Δt - 
                Q * up[2(m+1)*(n+1)+1:end]
        rhs2[injection] -= 1.0
        rhs2[production] += 1.0
        rhs = [rhs1;rhs2]        
        rhs[[bdnode; bdnode.+ (m+1)*(n+1)]] = bdval 
        err = norm(rhs)
        @info err
        if err<1e-8
            break 
        end

        A = [stiff -b*L'
            b*L/Δt 1/Δt*M-Q]
        A, _ = fem_impose_coupled_Dirichlet_boundary_condition(A, bdnode, m, n, h)
        Δu = A\rhs
        up -= Δu
        
    end

    global σ0, α0 = σ, α
    U[:,i+1] = up
end

visualize_displacement(U, m, n, h, name="_ppi3")
visualize_pressure(U, m, n, h, name="_ppi3")
