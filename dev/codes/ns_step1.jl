using Revise
using PoreFlow
using PyPlot
using ADCME


m = 20
n = 20
h = 1/n
dt = 0.1

ρ=1.0
μ=1.0
ν=μ/ρ

x, y = @vars x y
u = x*(1-x)*y*(1-y)
v = u 
ux = diff(u, x)
uy = diff(u, y)
uxx = diff(ux, x)
uyy = diff(uy, y)
p = u 
px = diff(p, x)
py = diff(p, y)

f = u/dt + 2u*diff(2u, x) + 2v*diff(2v, y) + 
    px - (2uxx + 2uyy)
g = u/dt + 2u*diff(2u, x) + 2v*diff(2v, y) + 
        py - (2uxx + 2uyy)
fcode = replace(replace(sympy.julia_code(f), ".*"=>"*"), ".^"=>"^")
gcode = replace(replace(sympy.julia_code(g), ".*"=>"*"), ".^"=>"^")


function func(x, y)
    2*x*y*(1 - x)*(1 - y)*(-2*x*y*(1 - x) + 2*x*(1 - x)*(1 - y)) + 2*x*y*(1 - x)*(1 - y)*(-2*x*y*(1 - y) + 2*y*(1 - x)*(1 - y)) + 10.0*x*y*(1 - x)*(1 - y) - x*y*(1 - y) + 4*x*(1 - x) + y*(1 - x)*(1 - y) + 4*y*(1 - y)
end

function gunc(x, y)
    2*x*y*(1 - x)*(1 - y)*(-2*x*y*(1 - x) + 2*x*(1 - x)*(1 - y)) + 2*x*y*(1 - x)*(1 - y)*(-2*x*y*(1 - y) + 2*y*(1 - x)*(1 - y)) + 10.0*x*y*(1 - x)*(1 - y) - x*y*(1 - x) + x*(1 - x)*(1 - y) + 4*x*(1 - x) + 4*y*(1 - y)
end




function step1(U, p0, Source = missing)
    Source = coalesce(Source, zeros(2*(m+1)*(n+1)))
    u0 = U[1:(m+1)*(n+1)]
    v0 = U[(m+1)*(n+1)+1:end]
    uxy = eval_grad_on_gauss_pts1(u0, m, n, h)
    vxy = eval_grad_on_gauss_pts1(v0, m, n, h)

    M1 = constant(compute_fem_mass_matrix1(m, n, h))
    M2 = compute_fem_mass_matrix1(uxy[:,1], m, n, h)
    M3 = compute_fem_advection_matrix1(u0, v0,
                m, n, h)
    M4 = constant(compute_fem_laplace_matrix1(m, n, h))
    A11 = M1 + M2 + M3 + M4

    A12 = compute_fem_mass_matrix1(uxy[:,2], m, n, h)

    A21 = compute_fem_mass_matrix1(vxy[:,1], m, n, h)

    M1 = constant(compute_fem_mass_matrix1(m, n, h))
    M2 = compute_fem_mass_matrix1(vxy[:,2], m, n, h)
    M3 = compute_fem_advection_matrix1(v0, u0, 
                m, n, h)
    M4 = constant(compute_fem_laplace_matrix1(m, n, h))
    A22 = M1 + M2 + M3 + M4

    A = [A11 A12
    A21 A22]


    grad_p_fem = compute_interaction_term(p0, m, n, h) 
    b1 = grad_p_fem 

    u0_ = fem_to_gauss_points(u0, m, n, h)
    v0_ = fem_to_gauss_points(v0, m, n, h)

    s1 = u0_ .* uxy[:,1] + v0_ .* uxy[:,2]
    s2 = u0_ .* vxy[:,1] + v0_ .* vxy[:,2]
    b2 = compute_fem_source_term(s1, s2, m, n, h)

    K = constant(compute_fem_laplace_matrix(m, n, h))
    lap_mat = μ / ρ * K

    F = Source + b1 - lap_mat * [u0;v0] - b2

    bd = bcnode("all", m, n, h)
    A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h)
    F = scatter_update(F, [bd; bd.+(m+1)*(n+1)], zeros(2length(bd)))
    op = tf.print("\nA=", Array(A), "\nF=", F, summarize=-1)
    F = bind(F, op)
    sol = A\F
    return sol
end



xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u0 = constant(@. x*(1-x)*y*(1-y))
v0 = constant(@. x*(1-x)*y*(1-y))
xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = constant(@. x*(1-x)*y*(1-y))


f1 = eval_f_on_gauss_pts(func, m, n, h)
f2 = eval_f_on_gauss_pts(gunc, m, n, h)
Source = compute_fem_source_term(f1, f2, m, n, h) 

sol = step1([u0;v0], p0, Source)
sess = Session(); init(sess)
S = run(sess, sol)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(S[1:(m+1)*(n+1)], m, n, h)
subplot(122)
visualize_scalar_on_fem_points(run(sess, u0), m, n, h)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(S[(m+1)*(n+1)+1:end], m, n, h)
subplot(122)
visualize_scalar_on_fem_points(run(sess, v0), m, n, h)



