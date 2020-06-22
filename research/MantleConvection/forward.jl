using Revise
using PoreFlow
using ADCME
using PyPlot
using Statistics
using LinearAlgebra
using ADCMEKit

m = 40
n = 20
h = 1/n
NT = 100
Δt = 1/NT 


u = 0.5*[ones(m*n);zeros(m*n)]

up_and_down = bcedge("upper|lower", m, n, h)

M = compute_fvm_mass_matrix(m, n, h)
K, rhs1 = compute_fvm_advection_matrix(u, up_and_down, zeros(size(up_and_down, 1)), m, n, h)
S, rhs2 = compute_fvm_tpfa_matrix(missing, up_and_down, zeros(size(up_and_down, 1)), m, n, h)

A = M/Δt + K - 0.01*S 

U = zeros(m*n, NT+1)
xy = fvm_nodes(m, n, h)
U[:,1] = @. exp( - 10 * ((xy[:,1]-1.0)^2 + (xy[:,2]-0.5)^2))

sess = Session(); init(sess)

A = run(sess, A)
for i = 1:NT 
    U[:, i+1] = A\(M*U[:,i]/Δt  - rhs2)
end
Z = zeros(NT+1, n, m)
for i = 1:NT+1
    Z[i,:,:] = reshape(U[:,i], m, n)'
end
p = visualize_scalar_on_fvm_points(Z, m, n, h)
saveanim(p, "advec.gif")

