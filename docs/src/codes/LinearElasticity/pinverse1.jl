using Revise
using PoreFlow
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

E = Variable(1.0)
ν = Variable(0.0)
D = compute_plane_stress_matrix(E*ones(get_ngauss(mmesh)), ν*ones(get_ngauss(mmesh)))
K = compute_fem_stiffness_matrix(D, mmesh)

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
U = matread("fenics/data1.mat")["u"]

using Random; Random.seed!(233)
idx = rand(1:mmesh.ndof, 100)
idx = [idx; idx .+ mmesh.ndof]

loss = sum((U[idx] - u[idx])^2) * 1e10
sess = Session(); init(sess)

loss_ = BFGS!(sess, loss)
run(sess, [E, ν])

