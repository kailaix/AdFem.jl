using Revise
using PoreFlow
using PyPlot

n = 20
mesh = Mesh(n, n, 1/n)
bd = bcedge(mesh)

# top edges 
bdedge = []
for i = 1:size(bd, 1)
    a, b = bd[i,:]
    if mesh.nodes[a,2]<1e-5 && mesh.nodes[b, 2]<1e-5
        push!(bdedge, [a b])
    end
end
bdedge = vcat(bdedge...)

bdnode_ = bcnode(mesh)
bdnode = Int64[]
for s in bdnode_
    if mesh.nodes[s, 1] > 1e-5 && mesh.nodes[s, 1] < 1-1e-5 && mesh.nodes[s, 2] < 1e-5
        continue 
    else
        push!(bdnode, s)
    end
end


F1 = eval_f_on_gauss_pts((x,y)->3.0, mesh)
F2 = eval_f_on_gauss_pts((x,y)->-1.0, mesh)
F1 = compute_fem_source_term1(F1, mesh)
F2 = compute_fem_source_term1(F2, mesh)
F = [F1;F2]


t1 = eval_f_on_boundary_edge((x,y)->-x-y, bdedge, mesh)
t2 = eval_f_on_boundary_edge((x,y)->2y, bdedge, mesh)
T1 = compute_fem_traction_term1(t1, bdedge, mesh)
T2 = compute_fem_traction_term1(t2, bdedge, mesh)
T = [T1;T2]

D = diagm(0=>[1,1,0.5])
K = constant(compute_fem_stiffness_matrix(D, mesh))
rhs = T - F 
bdval = [eval_f_on_boundary_node((x,y)->x^2+y^2, bdnode, mesh);
        eval_f_on_boundary_node((x,y)->x^2-y^2, bdnode, mesh)]
DOF = [bdnode;bdnode .+ mesh.ndof]
rhs[DOF] = bdval
K, Kbd = fem_impose_Dirichlet_boundary_condition1(K, DOF, mesh)
u = K\(rhs-Kbd*bdval)

sess = Session(); init(sess)
S = run(sess, u)


figure(figsize=[10,4])
subplot(121)
visualize_scalar_on_fem_points(S[1:mesh.nnode], mesh)
subplot(122)
visualize_scalar_on_fem_points(S[mesh.nnode+1:2mesh.nnode], mesh)
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
X = mesh.nodes[:,1]
Y = mesh.nodes[:,2]
U = (@. X^2+Y^2)
V = (@. X^2-Y^2)
subplot(121)
visualize_scalar_on_fem_points(U, mesh)
subplot(122)
visualize_scalar_on_fem_points(V, mesh)
savefig("exact.png")



figure(figsize=[10,4])
subplot(121)
X = mesh.nodes[:,1]
Y = mesh.nodes[:,2]
U = (@. X^2+Y^2)
V = (@. X^2-Y^2)
subplot(121)
visualize_scalar_on_fem_points(U-S[1:mesh.nnode], mesh)
subplot(122)
visualize_scalar_on_fem_points(V-S[mesh.nnode+1:2mesh.nnode], mesh)
savefig("difference.png")