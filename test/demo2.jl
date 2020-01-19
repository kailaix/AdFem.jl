# Example from: 
# Splitting schemes for poroelasticity and
# thermoelasticity problems
using Revise
using PoreFlow
using PyCall
np = pyimport("numpy")

# Domain information 
Δt = 0.1 * 24*60*60
NT = 30
n = 30
m = 2n 
h = 5.0/n 

G = 15e9 
λ = 10e9
pd = PoreData(
    M = 50e9,
    b = 1.0, 
    ρb = 2400., # ?
    ρf = 1000., # ?
    kp = 1e-18,
    E = G*(3λ+2G)/(λ+G),
    ν = λ/(2(λ+G)),
    μ = 0.001,
    Pi = 10e6,
    Bf = 1.0, # ?
    g = 0.0
)

injection = 10.01e6
production = 9.99e6
bdnode1 = Int64[]
bdnode2 = Int64[]
bdval1 = Float64[]
bdval2 = Float64[]
X = (0:m)*h
Y = (0:n)*h
X, Y = np.meshgrid(X,Y)
for i = 1:m+1
    j = n+1
    push!(bdnode1, i+(j-1)*(m+1))
    push!(bdnode2, i+(j-1)*(m+1)+(m+1)*(n+1))
    push!(bdval1, 0.0)
    push!(bdval2, 0.0)
end
for j=2:n
    i = 1
    push!(bdnode1, i+(j-1)*(m+1))
    push!(bdval1, 0.0)
    
    i = m+1
    push!(bdnode1, i+(j-1)*(m+1))
    push!(bdval1, 0.0)
end

bdnode = [bdnode1;bdnode2]
bdval = [bdval1;bdval2]

pval = Float64[]
pnode = Int64[]
for j = 1:3div(n,4)
    k = div(m,4)
    idx = k+(j-1)*m
    push!(pval, injection)
    push!(pnode, idx)

    k = 3div(m,4)
    idx = k+(j-1)*m
    push!(pval, production)
    push!(pnode, idx)
end

# Physical parameters
K = compute_elasticity_tangent(pd.E, pd.ν)
Q = compute_fluid_tpfa_matrix(m, n, h)
M = compute_fem_stiffness_matrix(K, m, n, h)
L = compute_interaction_matrix(m, n, h)
A = trim_coupled(pd, Q, L, M, bdnode, Δt, m, n, h)
A = coupled_impose_pressure(A, pnode, m, n, h)
U = zeros(m*n+2(m+1)*(n+1), NT+1)
U[2(m+1)*(n+1)+1:end, 1] .= pd.Pi 

for i = 1:NT 
    @info i
    rhs1 = zeros(2*(m+1)*(n+1))
    rhs2 = 1/pd.M/Δt * h^2 * U[2(m+1)*(n+1)+1:end, i] +
         pd.b/Δt * compute_fvm_mechanics_term(U[1:2(m+1)*(n+1), i],m,n,h) 
        
    rhs = [rhs1;rhs2]
    rhs[bdnode] = bdval
    rhs[pnode .+ 2*(m+1)*(n+1)] = pval 
    U[:,i+1] = A\rhs
end

# U[2(m+1)*(n+1)+1:end, :] .-= pd.Pi 
visualize_pressure(U, m, n, h)
visualize_displacement(U, m, n, h, scale=1e-7)
visualize_stress(K, U, m, n, h)