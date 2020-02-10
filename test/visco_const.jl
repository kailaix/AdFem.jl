using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
np = pyimport("numpy")

λ = 0.0
μ = 0.5
η = 0.1 # change η to +∞ for linear elasticity 

β = 1/4; γ = 1/2
a = b = 0.1
m = 40
n = 20
h = 0.01
NT = 200
Δt = 5/NT 
bdedge = []
for j = 1:n 
  push!(bdedge, [(j-1)*(m+1)+m+1 j*(m+1)+m+1])
end
bdedge = vcat(bdedge...)

bdnode = Int64[]
for j = 1:n+1
  push!(bdnode, (j-1)*(m+1)+1)
end

mvnode = Int64[]
for j = 1:n+1
  push!(mvnode, (j-1)*(m+1)+m+1)
end

G = [1/Δt+μ/η -μ/3η 0.0
  -μ/3η 1/Δt+μ/η-μ/3η 0.0
  0.0 0.0 1/Δt+μ/η]
S = [2μ/Δt+λ/Δt λ/Δt 0.0
    λ/Δt 2μ/Δt+λ/Δt 0.0
    0.0 0.0 μ/Δt]
invG = inv(G)
H = invG*S

M = compute_fem_mass_matrix1(m, n, h)
Zero = spzeros((m+1)*(n+1), (m+1)*(n+1))
M = [M Zero;Zero M]

K = compute_fem_stiffness_matrix(H, m, n, h)
C = a*M + b*K # damping matrix 

L = M + γ*Δt*C + β*Δt^2*K
L, Lbd = fem_impose_Dirichlet_boundary_condition(L, [bdnode;mvnode], m, n, h)

a = zeros(2(m+1)*(n+1))
v = zeros(2(m+1)*(n+1))
d = zeros(2(m+1)*(n+1))
U = zeros(2(m+1)*(n+1),NT+1)
Sigma = zeros(NT+1, 4m*n, 3)
Varepsilon = zeros(NT+1, 4m*n, 3)
for i = 1:NT 
    global a, v, d
    rhs = zeros(2*(m+1)*(n+1)) # no external force

    F = compute_strain_energy_term(Sigma[i,:,:]*invG/Δt, m, n, h) - K * U[:,i]
    rhs -= Δt^2 * F

    td = d + Δt*v + Δt^2/2*(1-2β)*a 

    tv = v + (1-γ)*Δt*a 
    tv[mvnode] .= 0.01
    tv[mvnode.+(m+1)*(n+1)] .= 0.0
    
    rhs = rhs - C*tv - K*td
    rhs[[bdnode; mvnode; bdnode.+(m+1)*(n+1); mvnode.+(m+1)*(n+1)]] .= 0.0


    a = L\rhs 
    d = td + β*Δt^2*a 
    v = tv + γ*Δt*a 

    v[mvnode] .= 0.01
    v[mvnode.+(m+1)*(n+1)] .= 0.0

    U[:,i+1] = d

    Varepsilon[i+1,:,:] = eval_strain_on_gauss_pts(U[:,i+1], m, n, h)
    Sigma[i+1,:,:] = Sigma[i,:,:]*invG/Δt +  (Varepsilon[i+1,:,:]-Varepsilon[i,:,:])*(invG*S)
end


# visualize_scattered_displacement(U, m, n, h; name = "_viscoelasticity", xlim_=[-0.01,0.5], ylim_=[-0.05,0.22])
# visualize_displacement(U, m, n, h;  name = "_viscoelasticity")
# visualize_stress(H, U, m, n, h;  name = "_viscoelasticity")

close("all")
figure(figsize=(15,5))
subplot(1,3,1)
idx = div(n,2)*(m+1) + m+1
plot((0:NT)*Δt, U[idx,:])
xlabel("time")
ylabel("\$u_x\$")

subplot(1,3,2)
idx = 4*(div(n,2)*m + m)
plot((0:NT)*Δt, Sigma[:,idx,1])
xlabel("time")
ylabel("\$\\sigma_{xx}\$")

subplot(1,3,3)
idx = 4*(div(n,2)*m + m)
plot((0:NT)*Δt, Varepsilon[:,idx,1])
xlabel("time")
ylabel("\$\\varepsilon_{xx}\$")
savefig("visco.png")
