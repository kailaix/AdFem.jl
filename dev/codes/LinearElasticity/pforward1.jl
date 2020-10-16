using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics
using MAT 

mmesh = Mesh(50, 50, 1/50, degree=2)

left = bcnode((x,y)->x<1e-5, mmesh)
right = bcedge((x1,y1,x2,y2)->(x1>0.049-1e-5) && (x2>0.049-1e-5), mmesh)

t1 = eval_f_on_boundary_edge((x,y)->1.0e-4, right, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->0.0, right, mmesh)
rhs = compute_fem_traction_term(t1, t2, right, mmesh)

E = 1.5
ν = 0.3
D = compute_plane_stress_matrix(E*ones(get_ngauss(mmesh)), ν*ones(get_ngauss(mmesh)))
K = compute_fem_stiffness_matrix(D, mmesh)

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
sess = Session(); init(sess)
S = run(sess, u)

matwrite("fenics/data1.mat", Dict("u"=>S))


close("all")
figure(figsize=(20, 5))
subplot(131)
visualize_scalar_on_fem_points(S[1:mmesh.nnode], mmesh)
title("x displacement")
subplot(132)
visualize_scalar_on_fem_points(S[1+mmesh.ndof:mmesh.nnode + mmesh.ndof], mmesh)
title("y displacement")
subplot(133)
Dval = run(sess, D)
visualize_von_mises_stress(Dval, S, mmesh)
title("von Mises Stress")
savefig("fenics/fwd1.png")