using AdFem
using PyPlot
using SparseArrays



function f1func(x,y)
    18.8495559215388*pi^2*sin(pi*x)^2*sin(pi*y)*cos(pi*y) - 6.28318530717959*pi^2*sin(pi*y)*cos(pi*x)^2*cos(pi*y) + pi*sin(pi*y)*cos(pi*x)
end
function f2func(x,y)
    -18.8495559215388*pi^2*sin(pi*x)*sin(pi*y)^2*cos(pi*x) + 6.28318530717959*pi^2*sin(pi*x)*cos(pi*x)*cos(pi*y)^2 + pi*sin(pi*x)*cos(pi*y)
end

m = 50
n = 50
h = 1/n
ν = 0.5
K = ν*constant(compute_fem_laplace_matrix(m, n, h))
B = constant(compute_interaction_matrix(m, n, h))
Z = [K -B'
-B spzero(size(B,1))]

bd = bcnode("all", m, n, h)
# Due to the rank insufficiency of bilinear quadrilateral element, 
# we add more pressure constraints 
bd = [bd; bd .+ (m+1)*(n+1); ((1:m) .+ 2(m+1)*(n+1))] 

F1 = eval_f_on_gauss_pts(f1func, m, n, h)
F2 = eval_f_on_gauss_pts(f2func, m, n, h)
F = compute_fem_source_term(F1, F2, m, n, h)
rhs = [F;zeros(m*n)]
Z, rhs = impose_Dirichlet_boundary_conditions(Z, rhs, bd, zeros(length(bd)))
sol = Z\rhs 

sess = Session(); init(sess)
S = run(sess, sol)

xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
U = @. 2*pi*sin(pi*x)*sin(pi*x)*cos(pi*y)*sin(pi*y)
figure(figsize=(12,5))
subplot(121)
visualize_scalar_on_fem_points(U, m, n, h)
title("Reference")
subplot(122)
visualize_scalar_on_fem_points(S[1:(m+1)*(n+1)], m, n, h)
title("Computed")
savefig("stokes1.png")

U = @. -2*pi*sin(pi*x)*sin(pi*y)*cos(pi*x)*sin(pi*y)
figure(figsize=(12,5))
subplot(121)
visualize_scalar_on_fem_points(U, m, n, h)
title("Reference")
subplot(122)
visualize_scalar_on_fem_points(S[(m+1)*(n+1)+1:2(m+1)*(n+1)], m, n, h)
title("Computed")
savefig("stokes2.png")


xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p = @. sin(pi*x)*sin(pi*y)
figure(figsize=(12,5))
subplot(121)
visualize_scalar_on_fvm_points(p, m, n, h)
title("Reference")
subplot(122)
visualize_scalar_on_fvm_points(S[2(m+1)*(n+1)+1:end], m, n, h)
title("Computed")
savefig("stokes3.png")