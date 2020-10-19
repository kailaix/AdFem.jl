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
Kr = 0.01*ones(m*n)
for i = 1:m 
    for j = div(n,3):div(n,3)*2
        Kr[(j-1)*m+i] = 10.0
    end
end

Q = compute_fvm_tpfa_matrix(Kr, m, n, h)
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
for i = 1:NT 
    t = i*Δt
    @info i
    f_ = eval_f_on_gauss_pts((x,y)->f(x,y,t), m, n, h)
    rhs1 = zeros(2(m+1)*(n+1))
    rhs2 =      b*L*U[1:2(m+1)*(n+1), i]/Δt + 
                M * U[2(m+1)*(n+1)+1:end,i]/Δt
    rhs2[injection] += 1.0
    rhs2[production] -= 1.0

    rhs = [rhs1;rhs2]

    bdval = zeros(2*length(bdnode))
    rhs[[bdnode; bdnode.+ (m+1)*(n+1)]] = bdval 
    rhs -= Abd * bdval 
    U[:,i+1] = A\rhs
end

visualize_displacement(U, m, n, h, name="_piheter")
visualize_pressure(U, m, n, h, name="_piheter")
