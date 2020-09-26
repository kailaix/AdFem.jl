using Revise
using PoreFlow
using PyPlot
using LinearAlgebra
using Statistics

mmesh = Mesh(50, 50, 1/50)

left = bcnode((x,y)->x<1e-5, mmesh)
right = bcedge((x1,y1,x2,y2)->(x1>0.049-1e-5) && (x2>0.049-1e-5), mmesh)

t1 = eval_f_on_boundary_edge((x,y)->1.0e-4, right, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->0.0, right, mmesh)
rhs = compute_fem_traction_term(t1, t2, right, mmesh)

D = diagm(0=>[1,1,0.5])
K = constant(compute_fem_stiffness_matrix(D, mmesh))

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
sess = Session(); init(sess)
S = run(sess, u)

# close("all")
# visualize_displacement(S*1e3, mmesh)
# savefig("test.png")

close("all")
figure(figsize=(10,20))
subplot(311)
visualize_scalar_on_fem_points(S[1:mmesh.nnode], mmesh)
subplot(312)
visualize_scalar_on_fem_points(S[1+mmesh.ndof:mmesh.nnode + mmesh.ndof], mmesh)
subplot(313)
visualize_von_mises_stress(D, S, mmesh)
savefig("test.png")