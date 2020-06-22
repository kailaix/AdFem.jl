# implicit scheme for advection-diffusion

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


hmat = zeros(4*m*n, 3, 3)
for i = 1:4*m*n 
    hmat[i,:,:] = diagm(0=>ones(3))
end 
hmat = constant(hmat)

hmat = constant(diagm(0=>ones(3)))

K = compute_fem_stiffness_matrix(hmat, m, n, h)
B = constant(compute_interaction_matrix(m, n, h))



Z = [K -B'
B spzero(size(B,1))]

bd = bcnode("lower", m, n, h)
Z, _ = fem_impose_Dirichlet_boundary_condition(Z, bd, m, n, h)


T = ones(4*m*n)
F1 = compute_fem_source_term1(T, m, n, h)
F0 = zero(F1)
rhs = [F1;F0;zeros(m*n)]

sol = Z[1:end-1, 1:end-1]\rhs[1:end-1]

sess = Session(); init(sess)
S = run(sess, sol)


# run(sess, K)