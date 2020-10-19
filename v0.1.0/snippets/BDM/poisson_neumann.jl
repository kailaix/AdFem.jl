# Solves the Poisson equation using the Mixed finite element method 
using Revise
using AdFem
using DelimitedFiles
using SparseArrays
using PyPlot


n = 50
mmesh = Mesh(n, n, 1/n, degree = BDM1)

function ufunc(x, y)
    x * (1-x) * y * (1-y)
end

function ffunc(x, y)
    -2x*(1-x) -2y*(1-y)
end

function gfunc(x, y)
    (1-2x)*y*(1-y)
end

A = compute_fem_bdm_mass_matrix1(mmesh)
B = compute_fem_bdm_div_matrix1(mmesh)
C = [A -B'
    -B spzeros(mmesh.nelem, mmesh.nelem)]

gD = (x1, y1, x2, y2)->!( (x1<1e-3 && y1>1e-3 && y1<1-1e-3) ||  (x2<1e-3 && y2>1e-3 && y2<1-1e-3))
gN = (x1, y1, x2, y2)->x1<1e-5 && x2<1e-5

D_bdedge = bcedge(gD, mmesh)
N_bdedge = bcedge(gN, mmesh)

t1 = eval_f_on_boundary_edge(ufunc, D_bdedge, mmesh)
g = compute_fem_traction_term1(t1, D_bdedge, mmesh) 
t2 = eval_f_on_gauss_pts(ffunc, mmesh)
f = compute_fvm_source_term(t2, mmesh)

gN = eval_f_on_boundary_edge(gfunc, N_bdedge, mmesh)
dof, val = impose_bdm_traction_boundary_condition1(gN, N_bdedge, mmesh)
rhs = [-g; f]
C, rhs = impose_Dirichlet_boundary_conditions(C, rhs, dof, val)

sol = C\rhs
u = sol[mmesh.ndof+1:end]
close("all")
figure(figsize=(15, 5))
subplot(131)
title("Reference")
xy = fvm_nodes(mmesh)
x, y = xy[:,1], xy[:,2]
uf = ufunc.(x, y)
visualize_scalar_on_fvm_points(uf, mmesh)
subplot(132)
title("Numerical")
visualize_scalar_on_fvm_points(u, mmesh)
subplot(133)
title("Absolute Error")
visualize_scalar_on_fvm_points( abs.(u - uf) , mmesh)
savefig("mixed_poisson_neumann.png")
