using Revise
using ADCME
using PoreFlow
using PyPlot
using SparseArrays

close("all")

# grid setup
m = 30
n = 30
h = 1/n
bc = bcedge("all", m, n, h)

# time step setup
t = 0;
t_final = 0.01;
NT = 20;
dt = t_final/NT;

# physical constants
# ρ = 1.2
# μ = 18 * 10^(-6)
ρ = 1
μ = 1
ν = μ / ρ

# exact solutions
function u1_exact(x1,x2,t)
    cos(2*pi*x1) * sin(2*pi*x2) * exp(-8*pi*pi*ν*t)
end

function u2_exact(x1,x2,t)
    -sin(2*pi*x1) * cos(2*pi*x2) * exp(-8*pi*pi*ν*t)
end

function p_exact(x1, x2, t, ρ)
    -ρ/4 * (cos(4*pi*x1) + cos(4*pi*x2)) * exp(-16*pi*pi*ν*t)
end

bdry = "Dirichlet" # impose Dirichlet boundary conditions on both velocity [u1, u2] and pressure p

function step1(U, p0, Source = missing)
    Source = coalesce(Source, zeros(2*(m+1)*(n+1)))
    u0 = U[1:(m+1)*(n+1)]
    v0 = U[(m+1)*(n+1)+1:end]
    u0_gauss = fem_to_gauss_points(u0, m, n, h)
    v0_gauss = fem_to_gauss_points(v0, m, n, h)

    gradu = eval_grad_on_gauss_pts1(u0, m, n, h) # du/dx = gradu[:,1], du/dy = gradu[:,2]
    gradv = eval_grad_on_gauss_pts1(v0, m, n, h) # dv/dx = gradv[:,1], dv/dy = gradv[:,2]

    M1 = constant(compute_fem_mass_matrix1(m, n, h))
    M2 = compute_fem_mass_matrix1(gradu[:,1], m, n, h)
    M3 = compute_fem_advection_matrix1(u0_gauss, v0_gauss, m, n, h)
    M4 = constant(compute_fem_laplace_matrix1(m, n, h))
    A11 = 1/dt * M1 + M2 + M3 + ν * M4

    A12 = compute_fem_mass_matrix1(gradu[:,2], m, n, h)

    A21 = compute_fem_mass_matrix1(gradv[:,1], m, n, h)

    M1 = constant(compute_fem_mass_matrix1(m, n, h))
    M2 = compute_fem_mass_matrix1(gradv[:,2], m, n, h)
    M3 = compute_fem_advection_matrix1(u0_gauss, v0_gauss, m, n, h)
    M4 = constant(compute_fem_laplace_matrix1(m, n, h))
    A22 = 1/dt * M1 + M2 + M3 + ν * M4

    A = [A11 A12
        A21 A22]

    grad_p = compute_interaction_term(p0, m, n, h) # [dp/dx; dp/dy] on fem points

    K = constant(compute_fem_laplace_matrix(m, n, h))

    s1 = u0_gauss .* gradu[:,1] + v0_gauss .* gradu[:,2]
    s2 = u0_gauss .* gradv[:,1] + v0_gauss .* gradv[:,2]
    b3 = compute_fem_source_term(s1, s2, m, n, h)

    F = Source + 1/ρ * grad_p - ν * K * [u0;v0] - b3

    if bdry == "Dirichlet"
        bd = bcnode("all", m, n, h)
        A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h)
        F = scatter_update(F, [bd; bd.+(m+1)*(n+1)], zeros(2length(bd)))
    end

    sol = A\F
    return sol
end

function step2(u0)
    B = constant(compute_interaction_matrix(m, n, h))
    bc = bcedge("all", m, n, h)
    A, _ = compute_fvm_tpfa_matrix(m, n, h)
    if bdry == "Dirichlet"
        A, _ = compute_fvm_tpfa_matrix(ones(m*n), bc, zeros(size(bc,1)), m, n, h)
    end
    A = constant(A)
    rhs = ρ / dt * B * u0
    sol = A\rhs
    return sol
end

function step3(u0, dp)
    A = constant(compute_fem_mass_matrix(m, n, h))
    bd = bcnode("all", m, n, h)

    grad_dp = - compute_interaction_term(dp, m, n, h)
    b = A * u0 - dt/ρ * grad_dp

    if bdry == "Dirichlet"
        A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h) # 2D boundary condition
        bd = [bd; (m+1)*(n+1) .+ bd]
        b = scatter_update(b, bd, zeros(length(bd)))
    end
    
    sol = A\b
    return sol
end


# input: U length 2(m+1)(n+1)
# input: p length mn
function solve_ns_one_step(U, p)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%$$$$$$$$$$   Step 1: solve for du, update U to U* (U_int)   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dU = step1(U, p)
    U_int = U + dU

   # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Step 2: solve Poisson equation for dp   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dp = step2(U_int)
    p_new = p + dp

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Step 3: solve for U_new   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_new = step3(U_int, dp)

    return U_new, p_new
end

function condition(i, velo_arr, p_arr)
    i <= NT + 1
end

function body(i, velo_arr, p_arr)
    velo = read(velo_arr, i-1)
    pres = read(p_arr, i-1)
    op = tf.print("i=",i)
    i = bind(i, op)
    velo_new, pres_new = solve_ns_one_step(velo, pres)
    velo_arr = write(velo_arr, i, velo_new)
    p_arr = write(p_arr, i, pres_new)
    return i+1, velo_arr, p_arr
end

# fem nodes
xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u0 = @.  u1_exact(x, y, 0.0)
v0 = @.  u2_exact(x, y, 0.0)

velo_arr = TensorArray(NT+1)
velo_arr = write(velo_arr, 1, [u0; v0])

# fvm nodes
xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @.  p_exact(x, y, 0.0, ρ)
p_arr = TensorArray(NT+1)
p_arr = write(p_arr, 1, p0)

i = constant(2, dtype=Int32)

_, velo, p = while_loop(condition, body, [i, velo_arr, p_arr])
velo = set_shape(stack(velo), (NT+1, 2*(m+1)*(n+1)))
p = set_shape(stack(p), (NT+1, m*n))

sess = Session(); init(sess)
output = run(sess, [velo, p])
out_v = output[1]
out_p = output[2]

dU = step1(constant([u0;v0]), constant(p0))
u_ = run(sess, dU)

# close("all")
# visualize_displacement(0.1*out_v, m, n, h)


# figure();visualize_scalar_on_points(u_[1:(m+1)*(n+1)], m, n, h)
# title("dU in first time step")
# # figure(); visualize_scalar_on_fvm_points(p0, m, n, h)

figure(figsize=(18,12))

subplot(231)
visualize_scalar_on_fem_points(out_v[1, 1:(1+m)*(1+n)], m, n, h)
title("initial velocity in x direction")

subplot(232)
visualize_scalar_on_fem_points(out_v[1, (1+m)*(1+n)+1:end], m, n, h)
title("initial velocity in y direction")

subplot(233)
visualize_scalar_on_fvm_points(out_p[1, :], m, n, h)
title("initial pressure")

subplot(234) 
visualize_scalar_on_fem_points(out_v[NT+1, 1:(1+m)*(1+n)], m, n, h)
title("final velocity in x direction")

subplot(235) 
visualize_scalar_on_fem_points(out_v[NT+1, (1+m)*(1+n)+1:end], m, n, h)
title("final velocity in y direction")

subplot(236)
visualize_scalar_on_fvm_points(out_p[NT+1, :], m, n, h)
title("final pressure")




# debug(sess, velo)
# U = @. 2*pi*sin(pi*x)*sin(pi*x)*cos(pi*y)*sin(pi*y)
# figure(figsize=(12,5))
# subplot(121)
# visualize_scalar_on_points(U, m, n, h)
# title("Reference")
# subplot(122)
# visualize_scalar_on_points(S[1:(m+1)*(n+1)], m, n, h)
# title("Computed")
# savefig("stokes1.png")

# U = @. -2*pi*sin(pi*x)*sin(pi*y)*cos(pi*x)*sin(pi*y)
# figure(figsize=(12,5))
# subplot(121)
# visualize_scalar_on_points(U, m, n, h)
# title("Reference")
# subplot(122)
# visualize_scalar_on_points(S[(m+1)*(n+1)+1:2(m+1)*(n+1)], m, n, h)
# title("Computed")
# savefig("stokes2.png")


# xy = fvm_nodes(m, n, h)
# x, y = xy[:,1], xy[:,2]
# p = @. sin(pi*x)*sin(pi*y)
# figure(figsize=(12,5))
# subplot(121)
# visualize_scalar_on_fvm_points(p, m, n, h)
# title("Reference")
# subplot(122)
# visualize_scalar_on_fvm_points(S[2(m+1)*(n+1)+1:end], m, n, h)
# title("Computed")