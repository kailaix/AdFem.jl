using Revise
using PoreFlow
using PyCall
np = pyimport("numpy")

# Domain information 
Δt = 1.
NT = 100
m = 40
n = 20
h = 0.1
traction = 2.125e6
production = 1.0e8
bdnode = Int64[]
bdval1 = Float64[]
bdval2 = Float64[]
X = (0:m)*h
Y = (0:n)*h
X, Y = np.meshgrid(X,Y)
for i = 2:m
    j = n+1
    push!(bdnode, i+(j-1)*(n+1))
    x = (i-1)*h
    y = (j-1)*h
    push!(bdval1, 0.0)
    push!(bdval2, 0.0)
end
bdnode = [bdnode;bdnode.+(m+1)*(n+1)]
bdval = [bdval1;bdval2]

bdedge = []
for i = 1:m 
    push!(bdedge, [i i+1])
end
for j = 1:n 
    push!(bdedge, [(j-1)*(m+1)+1 j*(m+1)+1])
    push!(bdedge, [(j-1)*(m+1)+m+1 j*(m+1)+m+1])
end
bdedge = vcat(bdedge...)

# Physical parameters
pd = PoreData()
K = compute_elasticity_tangent(pd.E, pd.ν)

# pd.b = 0.0
# formulate matrix
Q = compute_fluid_tpfa_matrix(m, n, h)
M = compute_fem_stiffness_matrix(K, m, n, h)
L = compute_interaction_matrix(m, n, h)
A = trim_coupled(pd, Q, L, M, bdnode, Δt, m, n, h)

U = zeros(m*n+2(m+1)*(n+1), NT+1)
U[2(m+1)*(n+1)+1:end, 1] .= pd.Pi 

productions = zeros(4m*n)
idx = div(n,2)*(m)+div(m,5)
productions[4(idx-1)+1:4idx] .= production

for i = 1:NT 
    rhs1 = compute_fem_normal_traction_term(traction, bdedge, m, n, h)
    rhs2 = compute_fvm_source_term(productions, m, n, h) +
        1/pd.M/Δt * h^2 * U[2(m+1)*(n+1)+1:end, i] +
         pd.b/Δt * compute_fvm_mechanics_term(U[1:2(m+1)*(n+1), i],m,n,h) 
        
    rhs = [rhs1;rhs2]
    rhs[bdnode] = bdval
    U[:,i+1] = A\rhs
end

U
close("all"); visualize_pressure(U, m, n, h); savefig("pressure.png")