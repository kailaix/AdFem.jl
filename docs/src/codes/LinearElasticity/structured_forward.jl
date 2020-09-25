using Revise
using PoreFlow
using LinearAlgebra
using PyPlot

m = 50
n = 50
h = 1/n

bdedge = bcedge("top", m, n, h)
bdnode = bcnode("left|right|bottom", m, n, h)

F1 = eval_f_on_gauss_pts((x,y)->3.0, m, n, h)
F2 = eval_f_on_gauss_pts((x,y)->-1.0, m, n, h)
F = compute_fem_source_term(F1, F2, m, n, h)

t1 = eval_f_on_boundary_edge((x,y)->-x-y, bdedge, m, n, h)
t2 = eval_f_on_boundary_edge((x,y)->2y, bdedge, m, n, h)
T = compute_fem_traction_term([t1 t2], bdedge, m, n, h)

D = constant(diagm(0=>[1,1,0.5]))
K = compute_fem_stiffness_matrix(D, m, n, h)
rhs = T - F 
bdval = [eval_f_on_boundary_node((x,y)->x^2+y^2, bdnode, m, n, h);
        eval_f_on_boundary_node((x,y)->x^2-y^2, bdnode, m, n, h)]
DOF = [bdnode;bdnode .+ (m+1)*(n+1)]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs
sess = Session(); init(sess)
S = run(sess, u)


figure(figsize=[10,4])
subplot(121)
visualize_scalar_on_fem_points(S[1:(m+1)*(n+1)], m, n, h)
subplot(122)
visualize_scalar_on_fem_points(S[(m+1)*(n+1)+1:2(m+1)*(n+1)], m, n, h)
savefig("numerical.png")

figure(figsize=[10,4])
x = LinRange(0, 1, 50)
X = zeros(50, 50)
Y = zeros(50, 50)
for i = 1:50
    for j = 1:50
        X[i,j] = x[i]
        Y[i,j] = x[j]
    end
end
subplot(121)
XY = fem_nodes(m, n, h)
X = XY[:,1]
Y = XY[:,2]
U = (@. X^2+Y^2)
V = (@. X^2-Y^2)
subplot(121)
visualize_scalar_on_fem_points(U, m, n, h)
subplot(122)
visualize_scalar_on_fem_points(V, m, n, h)
savefig("exact.png")



figure(figsize=[10,4])
subplot(121)
visualize_scalar_on_fem_points(U-S[1:(m+1)*(n+1)], m, n, h)
subplot(122)
visualize_scalar_on_fem_points(V-S[(m+1)*(n+1)+1:2(m+1)*(n+1)], m, n, h)
savefig("difference.png")