using Revise
using PoreFlow
using PyPlot

n = 20
mmesh = Mesh(n, n, 1/n, degree=2)
bdedge = bcedge((x1,y1,x2,y2)->((y1<1e-5) && (y2<1e-5)), mmesh)
bdnode = bcnode((x,y)->!(y<1e-5 && x>1e-5 && x<1-1e-5), mmesh)

F1 = eval_f_on_gauss_pts((x,y)->3.0, mmesh)
F2 = eval_f_on_gauss_pts((x,y)->-1.0, mmesh)
F = compute_fem_source_term(F1, F2, mmesh)

t1 = eval_f_on_boundary_edge((x,y)->-x-y, bdedge, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->2y, bdedge, mmesh)
T = compute_fem_traction_term(t1, t2, bdedge, mmesh)

D = constant(diagm(0=>[1,1,0.5]))
K = compute_fem_stiffness_matrix(D, mmesh)
rhs = T - F 
bdval = [eval_f_on_boundary_node((x,y)->x^2+y^2, bdnode, mmesh);
        eval_f_on_boundary_node((x,y)->x^2-y^2, bdnode, mmesh)]
DOF = [bdnode;bdnode .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
sess = Session(); init(sess)
S = run(sess, u)


figure(figsize=[10,4])
subplot(121)
visualize_scalar_on_fem_points(S[1:mmesh.nnode], mmesh)
subplot(122)
visualize_scalar_on_fem_points(S[mmesh.ndof+1:mmesh.ndof + mmesh.nnode], mmesh)
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
X = mmesh.nodes[:,1]
Y = mmesh.nodes[:,2]
U = (@. X^2+Y^2)
V = (@. X^2-Y^2)
subplot(121)
visualize_scalar_on_fem_points(U, mmesh)
subplot(122)
visualize_scalar_on_fem_points(V, mmesh)
savefig("exact.png")



figure(figsize=[10,4])
subplot(121)
visualize_scalar_on_fem_points(U-S[1:mmesh.nnode], mmesh)
subplot(122)
visualize_scalar_on_fem_points(V-S[mmesh.ndof+1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("difference.png")