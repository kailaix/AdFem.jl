# using SymPy 

# x, y = @vars x y
# u = x*(1-x)*y*(1-y)
# v = x*(1-x)*y*(1-y)^2

# p = x*(1-x)*y*(1-y)

# ux = diff(u,x)
# uy = diff(u,y)
# vx = diff(v,x)
# vy = diff(v,y)
# px = diff(p,x)
# py = diff(p,y)
# f = u*ux + v*uy - (diff(ux,x)+diff(uy,y)) + px
# g = u*vx + v*vy - (diff(vx,x)+diff(vy,y)) + py
# h = diff(u, x) + diff(v, y)
# println(replace(replace(sympy.julia_code(simplify(f)), ".^"=>"^"), ".*"=>"*"))
# println(replace(replace(sympy.julia_code(simplify(g)), ".^"=>"^"), ".*"=>"*"))
# println(replace(replace(sympy.julia_code(simplify(h)), ".^"=>"^"), ".*"=>"*"))

function u_exact(x,y)
    if y == 0.0
        return 1.0
    else
        return 0.0    
    end
end

function v_exact(x,y)
    0.0
end

function p_exact(x,y)
    0.0
end

function ffunc(x, y)
    #-x^2*y*(x - 1)^2*(y - 1)^2*(2*y - 1) + x*y^2*(x - 1)*(2*x - 1)*(y - 1)^2 + x*y*(y - 1) - 2*x*(x - 1) + y*(x - 1)*(y - 1) - 2*y*(y - 1)
    0.0
end
function gfunc(x, y)
    #x^2*y*(x - 1)^2*(y - 1)^3*(3*y - 1) - x*y^2*(x - 1)*(2*x - 1)*(y - 1)^3 + 3*x*y*(x - 1) + 5*x*(x - 1)*(y - 1) + 2*y*(y - 1)^2
    0.0
end

function hfunc(x,y)
    # (y - 1)*(-2*x*y*(x - 1) + x*y - x*(x - 1)*(y - 1) + y*(x - 1))
    0.0
end


using LinearAlgebra
using AdFem
using PyPlot
using SparseArrays

m = 100
n = 100
h = 1/n
nu = 0.01


F1 = compute_fem_source_term1(eval_f_on_gauss_pts(ffunc, m, n, h), m, n, h)
F2 = compute_fem_source_term1(eval_f_on_gauss_pts(gfunc, m, n, h), m, n, h)
H = h^2*eval_f_on_fvm_pts(hfunc, m, n, h)
B = constant(compute_interaction_matrix(m, n, h))

# compute F
Laplace = constant(nu*compute_fem_laplace_matrix1(m, n, h))
function compute_residual(S)
    u, v, p = S[1:(m+1)*(n+1)], S[(m+1)*(n+1)+1:2(m+1)*(n+1)], S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    interaction = compute_interaction_term(p, m, n, h) # julia kernel needed
    f1 = compute_fem_source_term1(ugauss.*ux, m, n, h)
    f2 = compute_fem_source_term1(vgauss.*uy, m, n, h)
    f3 = -interaction[1:(m+1)*(n+1)]
    f4 = Laplace*u 
    f5 = -F1
    F = f1 + f2 + f3 + f4 + f5 

    g1 = compute_fem_source_term1(ugauss.*vx, m, n, h)
    g2 = compute_fem_source_term1(vgauss.*vy, m, n, h)
    g3 = -interaction[(m+1)*(n+1)+1:end]    
    g4 = Laplace*v 
    g5 = -F2
    G = g1 + g2 + g3 + g4 + g5

    H0 = -B * [u;v] + H

    R = [F;G;H0]
    return R
end

function compute_jacobian(S)
    u, v, p = S[1:(m+1)*(n+1)], S[(m+1)*(n+1)+1:2(m+1)*(n+1)], S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    M1 = constant(compute_fem_mass_matrix1(ux, m, n, h))
    M2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h)) # a julia kernel needed
    M3 = Laplace
    Fu = M1 + M2 + M3 

    Fv = constant(compute_fem_mass_matrix1(uy, m, n, h))

    N1 = constant(compute_fem_mass_matrix1(vy, m, n, h))
    N2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h))
    N3 = Laplace
    Gv = N1 + N2 + N3 

    Gu = constant(compute_fem_mass_matrix1(vx, m, n, h))

    J0 = [Fu Fv
          Gu Gv]
    J = [J0 -B'
        -B spdiag(zeros(size(B,1)))]
end

NT = 5
# S = zeros(m*n+2(m+1)*(n+1), NT+1)
# S[1:m+1, 1] = ones(m+1,1)

bd = bcnode("all", m, n, h)
# bd = [bd; bd .+ (m+1)*(n+1); ((1:m) .+ 2(m+1)*(n+1))]
bd = [bd; bd .+ (m+1)*(n+1); 2*(m+1)*(n+1)+(n-1)*m+1:2*(m+1)*(n+1)+(n-1)*m+2] # only apply Dirichlet to velocity


function solve_steady_cavityflow_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, m, n, h)
    # residual[bd] .= 0.0
    residual = scatter_update(residual, bd, zeros(length(bd)))

    d = J\residual
    S_new = S - d
    return S_new
end


function condition(i, S_arr)
    i <= NT + 1
end

function body(i, S_arr)
    S = read(S_arr, i-1)
    op = tf.print("i=",i)
    i = bind(i, op)
    S_new = solve_steady_cavityflow_one_step(S)
    S_arr = write(S_arr, i, S_new)
    return i+1, S_arr
end

# for i = 1:NT 
#     residual = compute_residual(S[:,i])
#     J = compute_jacobian(S[:,i])
    
#     J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, m, n, h)
#     residual[bd] .= 0.0


#     d = J\residual
#     S[:,i+1] = S[:,i] - d
#     @info i, norm(residual)
# end


xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u0 = @. u_exact(x,y)
v0 = @. v_exact(x,y)

xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @. p_exact(x,y)

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, [u0; v0; p0])

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, 2*(m+1)*(n+1)+m*n))

sess = Session(); init(sess)
output = run(sess, S)
# S = output
# # out_v = output[:, 1:2*(m+1)*(n+1)]
# # out_p = output[:, 2*(m+1)*(n+1)+1:end]

# subplot(121)
# visualize_scalar_on_fem_points(output[NT+1, 1:(m+1)*(n+1)], m, n, h)
# subplot(122)
# visualize_scalar_on_fem_points(u, m, n, h)

figure(figsize=(20,10))
subplot(321)
visualize_scalar_on_fem_points(output[NT+1, 1:(m+1)*(n+1)], m, n, h)
subplot(322)
visualize_scalar_on_fem_points(u0, m, n, h)

subplot(323)
visualize_scalar_on_fem_points(output[NT+1, (m+1)*(n+1)+1:2*(m+1)*(n+1)], m, n, h)
subplot(324)
visualize_scalar_on_fem_points(v0, m, n, h)

subplot(325)
visualize_scalar_on_fvm_points(output[NT+1, 2*(m+1)*(n+1)+1:end], m, n, h)
subplot(326)
visualize_scalar_on_fvm_points(p0, m, n, h)
