using Revise
using PoreFlow
using LinearAlgebra
using PyPlot

m = 50
n = 50
h = 1/n

bdedge = []
for i = 1:m 
    push!(bdedge, [i i+1])
end
bdedge = vcat(bdedge...)

bdnode = Int64[]
for j = 1:n+1
    push!(bdnode, (j-1)*(m+1)+1)
    push!(bdnode, (j-1)*(m+1)+m+1)
end
for i = 2:m
    push!(bdnode, n*(m+1)+i)
end

F1 = eval_f_on_gauss_pts((x,y)->3.0, m, n, h)
F2 = eval_f_on_gauss_pts((x,y)->-1.0, m, n, h)
F = compute_fem_source_term(F1, F2, m, n, h)

t1 = eval_f_on_boundary_edge((x,y)->-x-y, bdedge, m, n, h)
t2 = eval_f_on_boundary_edge((x,y)->2y, bdedge, m, n, h)
T = compute_fem_traction_term([t1 t2], bdedge, m, n, h)

D = diagm(0=>[1,1,0.5])
K = compute_fem_stiffness_matrix(D, m, n, h)
rhs = T - F 
bdval = [eval_f_on_boundary_node((x,y)->x^2+y^2, bdnode, m, n, h);
        eval_f_on_boundary_node((x,y)->x^2-y^2, bdnode, m, n, h)]
rhs[[bdnode;bdnode .+ (m+1)*(n+1)]] = bdval
K, Kbd = fem_impose_Dirichlet_boundary_condition(K, bdnode, m, n, h)
u = K\(rhs-Kbd*bdval)
X, Y, U, V = plot_u(u, m, n, h)

figure(figsize=[10,4])
subplot(121)
pcolormesh(X, Y, (@. X^2+Y^2-U), alpha=0.6); xlabel("x"); ylabel("y"); title("Error for u")
colorbar()
subplot(122)
pcolormesh(X, Y, (@. X^2-Y^2-V), alpha=0.6); xlabel("x"); ylabel("y"); title("Error for v")
colorbar()
savefig("structured_test.png")