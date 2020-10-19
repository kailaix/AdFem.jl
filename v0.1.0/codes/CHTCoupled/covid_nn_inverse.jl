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
function nu_exact(x, y)
    (1 + 1 / (1 + x^2)) * 0.01
end

function nu_nn(x, y)
    # out = fc([x y], [20,20,20,1])^2 + 0.5 # N x 1 
    # out = fc([x y], [20,20,20,1])^2 + 0.01
    out = fc(x, [20,20,20,1])^2 + 0.01
    squeeze(out)
end

function u_exact(x,y)
    x*(1-x)*y*(1-y)
end

function v_exact(x,y)
    x*(1-x)*y*(1-y)
end

function p_exact(x,y)
    x*(1-x)*y*(1-y)
end

function t_exact(x,y)
    x*(1-x)*y*(1-y)
end

function k_exact(x,y)
    1.0
end

function ffunc_(x, y)
    # x*y*(1 - x)*(1 - y)*(-x*y*(1 - x) + x*(1 - x)*(1 - y)) + x*y*(1 - x)*(1 - y)*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - x*y*(1 - y) + 0.02*x*(1 - x) + y*(1 - x)*(1 - y) + 0.02*y*(1 - y)    # nu=0.01
    x*y*(1 - x)*(1 - y)*(-x*y*(1 - x) + x*(1 - x)*(1 - y)) + x*y*(1 - x)*(1 - y)*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - x*y*(1 - y) + y*(1 - x)*(1 - y) - (0.01 + 0.01/(x^2 + 1))*(-2*x*(1 - x) - 2*y*(1 - y))
end

function gfunc_(x, y)
    # x*y*(1 - x)*(1 - y)*(-x*y*(1 - x) + x*(1 - x)*(1 - y)) + x*y*(1 - x)*(1 - y)*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - x*y*(1 - x) + x*(1 - x)*(1 - y) + 0.02*x*(1 - x) + 0.02*y*(1 - y)    
    x*y*(1 - x)*(1 - y)*(-x*y*(1 - x) + x*(1 - x)*(1 - y)) + x*y*(1 - x)*(1 - y)*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - x*y*(1 - x) + x*(1 - x)*(1 - y) - (0.01 + 0.01/(x^2 + 1))*(-2*x*(1 - x) - 2*y*(1 - y))
end


function hfunc_(x,y)
    -x*y*(1 - x) - x*y*(1 - y) + x*(1 - x)*(1 - y) + y*(1 - x)*(1 - y)
end

function heat_source_func(x, y)
    x^2*y*(x - 1)^2*(y - 1)*(2*y - 1) + x*y^2*(x - 1)*(2*x - 1)*(y - 1)^2 - 2.0*x*(x - 1) - 2.0*y*(y - 1)
end

function q1_func_(t, x, y)
    -1.0*x*y*(1 - x)*(1 - y)
end


function q2_func_(t, x, y)
    -1.0*x*y*(1 - x)*(1 - y)
end

using LinearAlgebra
using MAT
using AdFem
using PyPlot
using SparseArrays

m = 20
n = 20
h = 1/n
# nu = 0.01
NT_transport = 100
Δt = 1/NT_transport
κ1 = 1.0 
κ2 = 1.0 

NT = 4    # number of iterations for Newton's method

bd = bcnode("all", m, n, h)
bd = [bd; bd .+ (m+1)*(n+1); 
    2*(m+1)*(n+1)+(n-1)*m+1:2*(m+1)*(n+1)+(n-1)*m+2;
    (2*(m+1)*(n+1)+m*n) .+ bd] 
# only apply Dirichlet to velocity; set left bottom two points to zero to fix rank deficient problem for pressure


## STEP 1: NAVIER STOKES EQUATIONS
F1 = compute_fem_source_term1(eval_f_on_gauss_pts(ffunc_, m, n, h), m, n, h)
F2 = compute_fem_source_term1(eval_f_on_gauss_pts(gfunc_, m, n, h), m, n, h)
H = h^2*eval_f_on_fvm_pts(hfunc_, m, n, h)
B = constant(compute_interaction_matrix(m, n, h))

# compute F
nu_gauss_exact = eval_f_on_gauss_pts(nu_exact, m, n, h)
xy = gauss_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
# nu_gauss = @. nu_nn(x, y); nu_gauss = stack(nu_gauss)
nu_gauss = nu_nn(x, y)
Laplace = compute_fem_laplace_matrix1(nu_gauss, m, n, h)
# Laplace = nu * constant(compute_fem_laplace_matrix1(m, n, h))
heat_source = eval_f_on_gauss_pts(heat_source_func, m, n, h)
heat_source = constant(compute_fem_source_term1(heat_source, m, n, h))
# The temperature term is 
# T_xx + T_yy + q = u T_x + v T_y

function compute_residual(S)
    u, v, p, T = S[1:(m+1)*(n+1)], 
        S[(m+1)*(n+1)+1:2(m+1)*(n+1)], 
        S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n],
        S[2(m+1)*(n+1)+m*n+1:end]
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

    # T_xx + T_yy = u T_x + v T_y - heat_source 
    # ugauss = zeros_like(ugauss)
    # vgauss = zeros_like(vgauss)
    A = constant(compute_fem_laplace_matrix1(m, n, h))
    T0 = A * T + compute_fem_advection_matrix1(ugauss,vgauss, m, n, h) * T - heat_source
    R = [F;G;H0;T0]
    return R
end

function compute_jacobian(S)
    u, v, p, T = S[1:(m+1)*(n+1)], 
                 S[(m+1)*(n+1)+1:2(m+1)*(n+1)], 
                 S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n],
                 S[2(m+1)*(n+1)+m*n+1:end]
        
    graduv = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = graduv[:,1,1], graduv[:,1,2], graduv[:,2,1], graduv[:,2,2]

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

    M = constant(compute_fem_laplace_matrix1(m, n, h)) +
         compute_fem_advection_matrix1(ugauss,vgauss, m, n, h) 

    gradT = eval_grad_on_gauss_pts1(T, m, n, h)
    Tx, Ty = gradT[:,1], gradT[:,2]
    DU_TX = constant(compute_fem_mass_matrix1(Tx, m, n, h))       # (m+1)*(n+1), (m+1)*(n+1)
    DV_TY = constant(compute_fem_mass_matrix1(Ty, m, n, h))       # (m+1)*(n+1), (m+1)*(n+1)
         
    J0 = [Fu Fv                         # Jacobian of [u; v]
          Gu Gv]

    J1 = [J0 -B'                        # Jacobian of [u; v; p]
          -B spdiag(zeros(size(B,1)))]

    N = 2*(m+1)*(n+1) + m*n                                     # length of [u; v; p]
    J = [J1 SparseTensor(spzeros(N,(m+1)*(n+1)))                # Jacobian of [u; v; p; T]
        DU_TX DV_TY SparseTensor(spzeros((m+1)*(n+1), m*n)) M]
end


function solve_steady_cavityflow_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, m, n, h)
    residual = scatter_update(residual, bd, zeros(length(bd)))    # residual[bd] .= 0.0 in Tensorflow syntax

    d = J\residual
    residual_norm = norm(residual)
    op = tf.print("residual norm", residual_norm)
    d = bind(d, op)
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

xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u0 = @. u_exact(x,y)
v0 = @. v_exact(x,y)
T0 = @. t_exact(x,y)

xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @. p_exact(x,y)

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, [u0; v0; p0; T0])
# S_arr = write(S_arr, 1, zeros(m*n+3*(m+1)*(n+1)))

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, 2*(m+1)*(n+1)+m*n+(m+1)*(n+1)))

u = S[NT+1, 1:(m+1)*(n+1)]
v = S[NT+1, (m+1)*(n+1)+1:2*(m+1)*(n+1)]
p = S[NT+1, 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n]
T = S[NT+1, 2*(m+1)*(n+1)+m*n+1:end]

## STEP 2: TRANSPORT EQUATION
# pre-compute source term
Q1 = zeros(NT_transport, (m+1)*(n+1))
Q2 = zeros(NT_transport, (m+1)*(n+1))
for i = 1:NT_transport
    t = i*Δt
    Q1[i, :] = eval_f_on_fem_pts((x,y)->q1_func_(t, x, y), m, n, h)
    Q2[i, :] = eval_f_on_fem_pts((x,y)->q2_func_(t, x, y), m, n, h)
end
Q1 = constant(Q1)
Q2 = constant(Q2)

function solve_transport_step(w1, w2, q1, q2)
    w1 = (1/Δt * w1 + κ1 * u + q1) / (1/Δt + κ1)
    w2 = (1/Δt * w2 + κ2 * v + q2) / (1/Δt + κ2)
    w1, w2
end

function transport_condition(i, w1_arr, w2_arr)
    i <= NT_transport
end

function transport_body(i, w1_arr, w2_arr)
    w1 = read(w1_arr, i)
    w2 = read(w2_arr, i)
    q1 = Q1[i]
    q2 = Q2[i]
    # op = tf.print("transport equation, step: ", i)
    # i = bind(i, op)
    w1, w2 = solve_transport_step(w1, w2, q1, q2)
    i+1, write(w1_arr, i+1, w1), write(w2_arr, i+1, w2)
end

function plot_velocity_pressure_viscosity(k)
    W1_computed, W2_computed = run(sess, [w1, w2])
    figure(figsize=(14,4));
    subplot(131)
    visualize_scalar_on_fem_points(W1_data[end, :], m, n, h); 
    title("Exact droplet x-velocity")
    subplot(132)
    visualize_scalar_on_fem_points(W1_computed[end, :], m, n, h);
    title("Computed droplet x-velocity")
    subplot(133)
    visualize_scalar_on_fem_points(W1_data[end, :] .- W1_computed[end, :], m, n, h); 
    title("Difference in droplet x-velocity")
    tight_layout()
    savefig("covid_figures5/covid_nn_w1_$k.png")

    figure(figsize=(14,4));
    subplot(131)
    visualize_scalar_on_fem_points(W2_data[end, :], m, n, h); 
    title("Exact droplet y-velocity")
    subplot(132)
    visualize_scalar_on_fem_points(W2_computed[end, :], m, n, h);
    title("Computed droplet y-velocity")
    subplot(133)
    visualize_scalar_on_fem_points(W2_data[end, :] .- W2_computed[end, :], m, n, h); 
    title("Difference in droplet y-velocity")
    tight_layout()
    savefig("covid_figures5/covid_nn_w2_$k.png")

    figure(figsize=(14,4));
    subplot(131)
    visualize_scalar_on_gauss_points(nu_gauss_exact, m, n, h); title("viscosity exact");gca().invert_yaxis()
    subplot(132)
    visualize_scalar_on_gauss_points(run(sess, nu_gauss), m, n, h); title("viscosity prediction");gca().invert_yaxis()
    subplot(133)
    visualize_scalar_on_gauss_points(nu_gauss_exact.-run(sess, nu_gauss), m, n, h); title("viscosity difference");gca().invert_yaxis()
    tight_layout()
    savefig("covid_figures5/covid_nn_visc$k.png")
end

i = constant(1, dtype = Int32)
w1_arr = TensorArray(NT_transport+1)
w2_arr = TensorArray(NT_transport+1)
w1_0 = eval_f_on_fem_pts((x,y)->x*y*(1-x)*(1-y), m, n, h)
# w2_0 = eval_f_on_fem_pts((x,y)->x*y*(1-x)*(1-y), m, n, h)
w2_0 = eval_f_on_fem_pts((x,y)->x^2*y^2*(1-x)*(1-y), m, n, h)
w1_arr = write(w1_arr, 1, w1_0)
w2_arr = write(w2_arr, 1, w2_0)
_, w1, w2 = while_loop(transport_condition, transport_body, [i, w1_arr, w2_arr])
w1, w2 = stack(w1), stack(w2)
w1 = set_shape(w1, (NT_transport+1, (m+1)*(n+1)))
w2 = set_shape(w2, (NT_transport+1, (m+1)*(n+1)))

W1_data = matread("covid_figures5/covid_fn_data.mat")["W1"]
W2_data = matread("covid_figures5/covid_fn_data.mat")["W2"]

sample_size = 22
idx = rand(1:(m+1)*(n+1), sample_size)

matwrite("covid_figures5/idx.mat", Dict(
        "idx"=>idx))

loss = mean((w1[:,idx].- W1_data[:,idx])^2) + mean((w2[:,idx].- W2_data[:,idx])^2)
loss = loss * 1e10

sess = Session(); init(sess)
@info run(sess, loss)
max_iter = 100

for k = 1:50
    loss_ = BFGS!(sess, loss, max_iter)
    matwrite("covid_figures5/loss$k.mat", Dict("L"=>loss_))
    close("all"); semilogy(loss_); title("loss vs. iteration")
    savefig("covid_figures5/nn_loss$k.png")
    plot_velocity_pressure_viscosity(k)
    ADCME.save(sess, "covid_figures5/nn$k.mat")
end

