using Revise
using ADCME
using AdFem
using PyPlot
using SparseArrays

close("all")

# grid setup
m = 100
n = 100
h = 1/n
bc = bcedge("all", m, n, h)
bd = bcnode("all", m, n, h)
bd_2d = [bd; (m+1)*(n+1) .+ bd]

# time step setup
t = 0;
t_final = 0.01;
NT = 1;
dt = t_final/NT;

# physical constants
ρ = 1
μ = 1
ν = μ / ρ

# pre-compute constant matrices
mass_mat = constant(compute_fem_mass_matrix1(m, n, h))
mass_mat_2d = constant(compute_fem_mass_matrix(m, n, h))
mass_mat_2d_bdry, _ = fem_impose_Dirichlet_boundary_condition(mass_mat_2d, bd, m, n, h) # 2D boundary condition
laplace_mat = constant(compute_fem_laplace_matrix1(m, n, h))
laplace_mat_2d = constant(compute_fem_laplace_matrix(m, n, h))
interact_mat = constant(compute_interaction_matrix(m, n, h))
tpfa_mat_bdry, _ = compute_fvm_tpfa_matrix(ones(m*n), bc, zeros(size(bc,1)), m, n, h)
tpfa_mat_bdry = constant(tpfa_mat_bdry)

# exact solutions
function u_exact(x,y,t)
    cos(2*pi*x) * sin(2*pi*y) * exp(-8*pi*pi*ν*t)
    # cos(x) * sin(y) * exp(-2*ν*t)
end

function v_exact(x,y,t)
    -sin(2*pi*x) * cos(2*pi*y) * exp(-8*pi*pi*ν*t)
    # -sin(x) * cos(y) * exp(-2*ν*t)
end

function p_exact(x, y, t, ρ)
    -ρ/4 * (cos(4*pi*x) + cos(4*pi*y)) * exp(-16*pi*pi*ν*t)
    # -ρ/4 * (cos(2*x1) + cos(2*x2)) * exp(-4*ν*t)
end

function step1(U, p0, Source = missing)
    Source = coalesce(Source, zeros(2*(m+1)*(n+1)))
    u0 = U[1:(m+1)*(n+1)]
    v0 = U[(m+1)*(n+1)+1:end]
    u0_gauss = fem_to_gauss_points(u0, m, n, h)
    v0_gauss = fem_to_gauss_points(v0, m, n, h)

    gradu = eval_grad_on_gauss_pts1(u0, m, n, h) # du/dx = gradu[:,1], du/dy = gradu[:,2]
    gradv = eval_grad_on_gauss_pts1(v0, m, n, h) # dv/dx = gradv[:,1], dv/dy = gradv[:,2]

    M1 = mass_mat
    M2 = compute_fem_mass_matrix1(gradu[:,1], m, n, h)
    M3 = compute_fem_advection_matrix1(u0_gauss, v0_gauss, m, n, h)
    M4 = laplace_mat
    A11 = 1/dt * M1 + M2 + M3 + ν * M4

    A12 = compute_fem_mass_matrix1(gradu[:,2], m, n, h)

    A21 = compute_fem_mass_matrix1(gradv[:,1], m, n, h)

    M2 = compute_fem_mass_matrix1(gradv[:,2], m, n, h) 
    A22 = 1/dt * M1 + M2 + M3 + ν * M4    # M1, M3, M4 are same as A11

    A = [A11 A12
        A21 A22]

    grad_p = compute_interaction_term(p0, m, n, h) # weak form of [dp/dx; dp/dy] on fem points

    s1 = u0_gauss .* gradu[:,1] + v0_gauss .* gradu[:,2]
    s2 = u0_gauss .* gradv[:,1] + v0_gauss .* gradv[:,2]
    b3 = compute_fem_source_term(s1, s2, m, n, h)

    F = Source + 1/ρ * grad_p - ν * laplace_mat_2d * [u0;v0] - b3

    A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h)
    F = scatter_update(F, bd_2d, zeros(length(bd_2d)))

    sol = A \ F
    return sol
end

function step2(U_int)
    rhs = ρ / dt * interact_mat * U_int
    sol = tpfa_mat_bdry \ rhs
    return sol
end

function step3(U_int, dp)
    grad_dp = - compute_interaction_term(dp, m, n, h)
    rhs = mass_mat_2d * U_int - dt/ρ * grad_dp
    rhs = scatter_update(rhs, bd_2d, zeros(length(bd_2d)))
    sol = mass_mat_2d_bdry \ rhs
    return sol
end

# input: U length 2(m+1)(n+1)
# input: p length mn
function solve_ns_one_step(U, p)
    dU = step1(U, p)
    U_int = U + dU
    dp = step2(U_int)
    p_new = p + dp
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
u0 = @.  u_exact(x, y, 0.0)
v0 = @.  v_exact(x, y, 0.0)
velo_arr = TensorArray(NT+1)
velo_arr = write(velo_arr, 1, [u0; v0])
# analytical solution
u_final = @.  u_exact(x, y, t_final)
v_final = @.  v_exact(x, y, t_final)

# fvm nodes
xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @.  p_exact(x, y, 0.0, ρ)
p_arr = TensorArray(NT+1)
p_arr = write(p_arr, 1, p0)
# analytical solution
p_final = @.  p_exact(x, y, t_final, ρ)


i = constant(2, dtype=Int32)

_, velo, p = while_loop(condition, body, [i, velo_arr, p_arr])
velo = set_shape(stack(velo), (NT+1, 2*(m+1)*(n+1)))
p = set_shape(stack(p), (NT+1, m*n))

sess = Session(); init(sess)
output = run(sess, [velo, p])
out_v = output[1]
out_p = output[2]



# dU = step1(constant([u0;v0]), constant(p0))
# u_ = run(sess, dU)

# close("all")
# visualize_displacement(0.1*out_v, m, n, h)


# figure();visualize_scalar_on_points(u_[1:(m+1)*(n+1)], m, n, h)
# title("dU in first time step")
# # figure(); visualize_scalar_on_fvm_points(p0, m, n, h)

figure(figsize=(18,12))

subplot(231)
visualize_scalar_on_fem_points(out_v[1, 1:(1+m)*(1+n)], m, n, h)
title("initial velocity in x direction")
# savefig("nsforward_initial_velocity_x.png")

subplot(232)
visualize_scalar_on_fem_points(out_v[1, (1+m)*(1+n)+1:end], m, n, h)
title("initial velocity in y direction")
# savefig("nsforward_initial_velocity_y.png")

subplot(233)
visualize_scalar_on_fvm_points(out_p[1, :], m, n, h)
title("initial pressure")
# savefig("nsforward_initial_pressure.png")

subplot(234) 
visualize_scalar_on_fem_points(out_v[NT+1, 1:(1+m)*(1+n)], m, n, h)
title("final velocity in x direction")
# savefig("nsforward_final_velocity_x_computed.png")

subplot(235) 
visualize_scalar_on_fem_points(out_v[NT+1, (1+m)*(1+n)+1:end], m, n, h)
title("final velocity in y direction")
# savefig("nsforward_final_velocity_y_computed.png")

subplot(236)
visualize_scalar_on_fvm_points(out_p[NT+1, :], m, n, h)
title("final pressure")
# savefig("nsforward_final_pressure_computed.png")

figure(); visualize_scalar_on_fem_points(u_final, m, n, h)

figure(); visualize_scalar_on_fem_points(u_final, m, n, h)

figure(); visualize_scalar_on_fvm_points(p_final, m, n, h)

