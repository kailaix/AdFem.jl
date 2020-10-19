using AdFem
using PyPlot

function g_func(x, y)
    -2*x*(1-x) - 2*y*(1-y) + 2*(1-2x)*y*(1-y) + 3*(1-2y)*x*(1-x)
end

m = 30
n = 30
h = 1/n

xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u = @. (x*(1-x)*y*(1-y))

A = constant(compute_fem_laplace_matrix1(m, n, h))
u0 = 2ones(2(m+1)*(n+1))
v0 = 3ones(2(m+1)*(n+1))
u0 = fem_to_gauss_points(u0, m, n, h)
v0 = fem_to_gauss_points(v0, m, n, h)
B = compute_fem_advection_matrix1(constant(u0), constant(v0), m, n, h)
L = -A+B
g = eval_f_on_gauss_pts(g_func, m, n, h)
rhs = compute_fem_source_term1(g, m, n, h)
bd = bcnode("all", m, n, h)
L, _ = fem_impose_Dirichlet_boundary_condition1(L, bd, m, n, h)
rhs[bd] .= 0.0
sol = L\rhs

sess = Session(); init(sess)
S = run(sess, sol)
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(S, m, n, h)
title("computed solution")
subplot(122)
visualize_scalar_on_fem_points(u, m, n, h)
title("exact solution")

# subplot(133)
# visualize_scalar_on_fem_points(S-u, m, n, h)
# title("difference between computed and exact")
