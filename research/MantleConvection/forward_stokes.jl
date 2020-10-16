# implicit scheme for advection-diffusion

using Revise
using AdFem
using ADCME
using PyPlot
using Statistics
using LinearAlgebra
using ADCMEKit

m = 100
n = 100
h = 1/n

function f1func(x,y)
    4.0*x^3*y^3 - 4.0*x^3*y*(1 - y^2) - 4.0*x*y^3*(1 - x^2) + 4.0*x*y*(1 - x^2)*(1 - y^2) - 2.0*x*(1 - x) - 4*y*(1 - y) - 1
end
function f2func(x,y)
    -20*x^2*y^2*(1 - x^2) - 10.0*x^2*y^2*(1 - y^2) + 4*x^2*(1 - x^2)*(1 - y^2) + 1.0*x*y - x*(1 - y) + 2.0*y^2*(1 - x^2)*(1 - y^2) - 1.0*y*(1 - x) + (1 - x)*(1 - y) - 1
end

function gfunc(x,y)
    -2*x^2*y^3*(1 - x^2) + 2*x^2*y*(1 - x^2)*(1 - y^2) - x*y*(1 - y) + y*(1 - x)*(1 - y)
end

hmat = zeros(4*m*n, 3, 3)
for i = 1:4*m*n 
    hmat[i,:,:] = 2diagm(0=>ones(3))
end 
hmat = constant(hmat)

K = compute_fem_stiffness_matrix(hmat, m, n, h)
B = constant(compute_interaction_matrix(m, n, h))

Z = [K -B'
B spzero(size(B,1))]

bd = bcnode("all", m, n, h)
Z, _ = fem_impose_Dirichlet_boundary_condition(Z, bd, m, n, h)


F1 = eval_f_on_gauss_pts(f1func, m, n, h)
F2 = eval_f_on_gauss_pts(f2func, m, n, h)
F = compute_fem_source_term(F1, F2, m, n, h)
xy = fvm_nodes(m, n, h)
G = gfunc.(xy[:,1], xy[:,2])
G = compute_fvm_source_term(G, m, n, h)
G = G .- mean(G)
rhs = [-F;G]
rhs[[bd;bd.+(m+1)*(n+1)]] .= 0.0

sol = Z[1:end-1, 1:end-1]\rhs[1:end-1]

sess = Session(); init(sess)
S = run(sess, sol)

# visualize_displacement(S[1:2(m+1)*(n+1)],m, n,h)
xy = fem_nodes(m, n, h)
U = @. (1-xy[:,1])*xy[:,1]*(1-xy[:,2])*xy[:,2]
figure(figsize=(12,5))
subplot(121)
visualize_scalar_on_fem_points(U, m, n, h)
title("Reference")
subplot(122)
visualize_scalar_on_fem_points(S[1:(m+1)*(n+1)], m, n, h)
title("Computed")
savefig("stokes1.png")

U = @. (1-xy[:,1]^2)*xy[:,1]^2*(1-xy[:,2]^2)*xy[:,2]^2
figure(figsize=(12,5))
subplot(121)
visualize_scalar_on_fem_points(U, m, n, h)
title("Reference")
subplot(122)
visualize_scalar_on_fem_points(S[(m+1)*(n+1)+1:2(m+1)*(n+1)], m, n, h)
title("Computed")
savefig("stokes2.png")
