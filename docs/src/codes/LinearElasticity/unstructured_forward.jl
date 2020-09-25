using Revise
using PoreFlow
using PyPlot

n = 20
mmesh = Mesh(n, n, 1/n, degree=2)
bd = bcedge(mmesh)

# top edges 
bdedge = []
for i = 1:size(bd, 1)
    a, b = bd[i,:]
    if mmesh.nodes[a,2]<1e-5 && mmesh.nodes[b, 2]<1e-5
        push!(bdedge, [a b])
    end
end
bdedge = vcat(bdedge...)

bdnode_ = bcnode(mmesh)
bdnode = Int64[]
for s in bdnode_
    if s>mmesh.nnode
        x = mmesh.nodes[mmesh.edges[s - mmesh.nnode,1], :]
        y = mmesh.nodes[mmesh.edges[s - mmesh.nnode,2], :]
        mid = (x+y)/2
        if mid[1] > 1e-5 && mid[1] < 1-1e-5 && mid[2] < 1e-5
            continue 
        continue 
            continue 
        else
            push!(bdnode, s)
        end
    else
        if mmesh.nodes[s, 1] > 1e-5 && mmesh.nodes[s, 1] < 1-1e-5 && mmesh.nodes[s, 2] < 1e-5
            continue 
        continue 
            continue 
        else
            push!(bdnode, s)
        end
    end
end


F1 = eval_f_on_gauss_pts((x,y)->3.0, mmesh)
F2 = eval_f_on_gauss_pts((x,y)->-1.0, mmesh)
F = compute_fem_source_term(F1, F2, mmesh)

t1 = eval_f_on_boundary_edge((x,y)->-x-y, bdedge, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->2y, bdedge, mmesh)
T = compute_fem_traction_term(t1, t2, bdedge, mmesh)

D = diagm(0=>[1,1,0.5])
K = constant(compute_fem_stiffness_matrix(D, mmesh))
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
X = mmesh.nodes[:,1]
Y = mmesh.nodes[:,2]
U = (@. X^2+Y^2)
V = (@. X^2-Y^2)
subplot(121)
visualize_scalar_on_fem_points(U-S[1:mmesh.nnode], mmesh)
subplot(122)
visualize_scalar_on_fem_points(V-S[mmesh.ndof+1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("difference.png")