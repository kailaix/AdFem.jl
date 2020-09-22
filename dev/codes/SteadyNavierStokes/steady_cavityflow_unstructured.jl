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
using MAT
using PoreFlow
using PyPlot
using SparseArrays

m = 20
n = 20
h = 1/n
mesh = Mesh(m, n, h)

# filename = "CHT_2D.stl"
# file_format = "stl"
# mesh = Mesh(filename, file_format = file_format)
# mesh = Mesh(mesh.nodes ./ 0.0305, mesh.elems)

nnode = size(mesh.nodes, 1)
nelem = size(mesh.elems, 1)
nu = 0.01

bd = Array{Int64, 1}([])
eps = 1e-6
for j = 1:nnode
    nodex, nodey = mesh.nodes[j, 1], mesh.nodes[j, 2]
    if abs(nodex-0.0) <= eps || abs(nodex-1.0) <= eps || abs(nodey-0.0) <= eps || abs(nodey-1.0) <= eps
        global bd = [bd; j]
    end
end

F1 = compute_fem_source_term1(constant(eval_f_on_gauss_pts(ffunc, mesh)), mesh)
F2 = compute_fem_source_term1(constant(eval_f_on_gauss_pts(gfunc, mesh)), mesh)
H = get_area(mesh) .* eval_f_on_fvm_pts(hfunc, mesh)
B = constant(compute_interaction_matrix(mesh))

# compute F
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(nnode)), mesh))
function compute_residual(S)
    u, v, p = S[1:nnode], S[nnode+1:2*nnode], S[2*nnode+1:2*nnode+nelem]
    ugauss = fem_to_gauss_points(u, mesh)
    vgauss = fem_to_gauss_points(v, mesh)
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)
    ux, uy, vx, vy = grad_u[:,1], grad_u[:,2], grad_v[:,1], grad_v[:,2]

    interaction = compute_interaction_term(p, mesh) # julia kernel needed
    f1 = compute_fem_source_term1(ugauss.*ux, mesh)
    f2 = compute_fem_source_term1(vgauss.*uy, mesh)
    f3 = -interaction[1:nnode]
    f4 = Laplace*u 
    f5 = -F1
    F = f1 + f2 + f3 + f4 + f5 

    g1 = compute_fem_source_term1(ugauss.*vx, mesh)
    g2 = compute_fem_source_term1(vgauss.*vy, mesh)
    g3 = -interaction[nnode+1:end]    
    g4 = Laplace*v 
    g5 = -F2
    G = g1 + g2 + g3 + g4 + g5

    H0 = -B * [u;v] + H

    R = [F;G;H0]
    return R
end

function compute_jacobian(S)
    u, v, p = S[1:nnode], S[nnode+1:2*nnode], S[2*nnode+1:2*nnode+nelem]
    ugauss = fem_to_gauss_points(u, mesh)
    vgauss = fem_to_gauss_points(v, mesh)
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)
    ux, uy, vx, vy = grad_u[:,1], grad_u[:,2], grad_v[:,1], grad_v[:,2]

    M1 = constant(compute_fem_mass_matrix1(ux, mesh))
    M2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), mesh)) # a julia kernel needed
    M3 = Laplace
    Fu = M1 + M2 + M3 

    Fv = constant(compute_fem_mass_matrix1(uy, mesh))

    N1 = constant(compute_fem_mass_matrix1(vy, mesh))
    N2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), mesh))
    N3 = Laplace
    Gv = N1 + N2 + N3 

    Gu = constant(compute_fem_mass_matrix1(vx, mesh))

    J0 = [Fu Fv
          Gu Gv]
    J = [J0 -B'
        -B spdiag(zeros(size(B,1)))]
end

NT = 5    # number of iterations for Newton's method

bd = [bd; bd .+ nnode; 2*nnode+nelem-10:2*nnode+nelem] 
# only apply Dirichlet to velocity; set last two elements to zero to fix rank deficient problem for pressure


function solve_steady_cavityflow_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, mesh)
    residual = scatter_update(residual, bd, zeros(length(bd)))    # residual[bd] .= 0.0 in Tensorflow syntax

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

xy = mesh.nodes
x, y = xy[:,1], xy[:,2]
u0 = @. u_exact(x,y)
v0 = @. v_exact(x,y)

xy = fvm_nodes(mesh) ### TODO: FVM_NODES(MESH)
x, y = xy[:,1], xy[:,2]
p0 = @. p_exact(x,y)

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, [u0; v0; p0])

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, 2*nnode+nelem))

sess = Session(); init(sess)
output = run(sess, S)

u_out, v_out, p_out = output[NT+1,:nnode], output[NT+1,nnode+1:2*nnode], output[NT+1,2*nnode+1:2*nnode+nelem]

# matwrite("steady_cavityflow_param_data.mat", 
#     Dict(
#         "V"=>output[end, 1:2*nnode]
#     ))

figure(figsize=(20,10))
subplot(321)
visualize_scalar_on_fem_points(u_out, m, n, h)
subplot(322)
visualize_scalar_on_fem_points(u0, m, n, h)

subplot(323)
visualize_scalar_on_fem_points(v_out, m, n, h)
subplot(324)
visualize_scalar_on_fem_points(v0, m, n, h)

subplot(325)
visualize_scalar_on_fvm_points(p_out, m, n, h)
subplot(326)
visualize_scalar_on_fvm_points(p0, m, n, h)

savefig(ns_unstructured_forward_solution.png)

# final_u=output[NT+1, 1:(1+m)*(1+n)]
# final_v=output[NT+1, (1+m)*(1+n)+1:2*(m+1)*(n+1)]
# final_p=output[NT+1, 2*(m+1)*(n+1)+1:end]

# u1 = final_u[50*101+1: 50*101+101]
# u2 = final_u[51:101:end]

# v1 = final_v[50*101+1: 50*101+101]
# v2 = final_v[51:101:end]
# xx = 0:0.01:1

# figure();plot(xx, u1);
# savefig("u_horizontal.png")

# figure();plot(xx, u2);
# savefig("u_vertical.png")

# figure();plot(xx, v1);
# savefig("v_horizontal.png")

# figure();plot(xx, v2);
# savefig("v_vertical.png")

# p1 = final_p[49*100+1: 49*100+100]
# p2 = final_p[50*100+1: 50*100+100]
# p3 = 0.5 * (p1 .+ p2)

# p4 = final_p[50:100:end]
# p5 = final_p[51:100:end]
# p6 = 0.5 * (p4 .+ p5)

# xx = 0.005:0.01:1

# figure();plot(xx, p3);
# savefig("p_horizontal.png")

# figure();plot(xx, p6);
# savefig("p_vertical.png")