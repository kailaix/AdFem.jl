using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics
using MAT 

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))
mmesh = Mesh(mmesh.nodes * 10, mmesh.elems, -1, 2)

ν = constant(0.001)
K = ν*constant(compute_fem_laplace_matrix(mmesh))
B = constant(compute_interaction_matrix(mmesh))
Z = [K -B'
    -B spzero(size(B,1))]

bd1 = bcnode((x,y)->y>0.23-1e-5, mmesh)
bd2 = bcnode((x,y)->y<=0.23-1e-5, mmesh)
bd = [bd1;bd2]
bd = [bd; bd .+ mmesh.ndof; 2mmesh.ndof + 1]


rhs = zeros(2mmesh.ndof + mmesh.nelem)
bdval = zeros(length(bd))
bdval[1:length(bd1)] .= 1.0
Z, rhs = impose_Dirichlet_boundary_conditions(Z, rhs, bd, bdval)
sol = Z\rhs 

sess = Session(); init(sess)
S = run(sess, sol)

matwrite("fenics/fwd1.mat", Dict("u"=>S))


close("all")
figure(figsize=(20, 5))
subplot(131)
visualize_scalar_on_fem_points(S[1:mmesh.nnode], mmesh)
title("x displacement")
subplot(132)
visualize_scalar_on_fem_points(S[1+mmesh.ndof:mmesh.nnode + mmesh.ndof], mmesh)
title("y displacement")
subplot(133)
visualize_scalar_on_fvm_points(S[2mmesh.ndof+1:end], mmesh)
title("Pressure")
savefig("fenics/fwd1.png")

close("all")
visualize_vector_on_fem_points(S[1:mmesh.nnode], S[1+mmesh.ndof:mmesh.nnode + mmesh.ndof], mmesh)
axis("scaled")
savefig("fenics/fwd1_quiver.png")