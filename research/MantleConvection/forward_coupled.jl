# implicit scheme for advection-diffusion

using Revise
using AdFem
using ADCME
using PyPlot
using Statistics
using LinearAlgebra
using ADCMEKit

m = 40
n = 40
h = 1/n
NT = 20
Δt = 0.1/NT 
Ra = 1e6

function solve_stokes(T, η)
    η = convert_to_tensor(η)
    K = compute_fem_laplace_matrix1(η, m, n, h)
    Z = spdiag(zeros((m+1)*(n+1)))
    K = [K Z; Z K]
    B = constant(compute_interaction_matrix(m, n, h))
    Z = [K -B'
    -B spzero(size(B,1))]
    bd = bcnode("upper|lower", m, n, h)
    bd = [bd; bd .+ (m+1)*(n+1)]
    bd_T = [1:m; (n-1)*m .+ (1:m)] # pressure up and lower 
    Z, _ = fem_impose_Dirichlet_boundary_condition1(Z, [bd;bd_T], m, n, h)
    T = reshape(repeat(T, 1, 4), (-1,))
    rhs = Ra * compute_fem_source_term(constant(zeros(4*m*n)), T, m, n, h)
    rhs = [rhs;constant(zeros(m*n))]
    rhs = scatter_update(rhs, [bd; bd_T], zeros(length(bd) + length(bd_T)) )
    sol = Z\rhs
    sol[1:2*(m+1)*(n+1)], sol[2*(m+1)*(n+1)+1:end]
end

function solve_heat_eq(u, T)
    up_and_down = bcedge("upper|lower", m, n, h)
    M = compute_fvm_mass_matrix(m, n, h)
    u_on_fvm_grid = [fem_to_fvm(u[1:(m+1)*(n+1)], m, n, h); fem_to_fvm(u[(m+1)*(n+1)+1:end], m, n, h)]
    K, rhs1 = compute_fvm_advection_matrix(u_on_fvm_grid, up_and_down, zeros(size(up_and_down, 1)), m, n, h)
    S, rhs2 = compute_fvm_tpfa_matrix(missing, up_and_down, zeros(size(up_and_down, 1)), m, n, h)
    A = M/Δt + K - S 
    T_new = A\(M*T/Δt - rhs1 + rhs2)
end 

function compute_viscosity_parameter(u, T; kwargs...)
    η = mantle_viscosity(u, T, m, n, h; kwargs...)
    return η
end

xy = fvm_nodes(m, n, h)
T0 = @. -exp(-100*((xy[:,1]-0.5)^2 + (xy[:,2]-0.2)^2))

u_arr = TensorArray(NT+1)
T_arr = TensorArray(NT+1)
p_arr = TensorArray(NT+1)
η_arr = TensorArray(NT+1)
i = constant(1, dtype = Int32)
u_arr = write(u_arr, 1, zeros(2*(m+1)*(n+1)))
T_arr = write(T_arr, 1, T0)
η_arr = write(η_arr, 1, zeros(4*m*n))
p_arr = write(p_arr, 1, zeros(m*n))

function condition(i, tas...)
    i<=NT 
end
function body(i, tas...)
    u_arr, T_arr, p_arr, _ = tas
    u, T, p = read(u_arr, i), read(T_arr, i), read(p_arr, i)
    T = solve_heat_eq(u, T)
    η = compute_viscosity_parameter(u, T)
    u, p = solve_stokes(T, η)
    op = tf.print(i)
    i = bind(i, op)
    i+1, write(u_arr, i+1, u), write(T_arr, i+1, T), write(p_arr, i+1, p), write(η_arr, i+1, η)
end

_, u, T, p, η = while_loop(condition, body, [i, u_arr, T_arr, p_arr, η_arr])
u = set_shape(stack(u), (NT+1, 2(m+1)*(n+1)))
T = set_shape(stack(T), (NT+1, m*n))
p = set_shape(stack(p), (NT+1, m*n))
η = set_shape(stack(η), (NT+1, 4m*n))
sess = Session(); init(sess)
u_, T_, p_, η_ = run(sess, [u, T, p, η])


# p = visualize_displacement(u_, m, n, h)
# saveanim(p, "displacement.gif")
# visualize_scalar_on_fvm_points(T_, m, n, h)
# visualize_scalar_on_fvm_points(p_, m, n, h)

visualize_scalar_on_fvm_points(T_, m, n, h)
# visualize_scalar_on_gauss_points(η_, m, n, h)