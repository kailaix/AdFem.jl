# implicit scheme for advection-diffusion

using Revise
using PoreFlow
using ADCME
using PyPlot
using Statistics
using LinearAlgebra
using ADCMEKit

m = 40
n = 20
h = 1/n
NT = 100
Δt = 1/NT 
Ra = 1.0

function solve_stokes(T, η)
    η = convert_to_tensor(η)
    z = constant(zeros(4m*n))
    hmat = reshape([η z z z η z z z η], (4m*n,3,3))
    K = compute_fem_stiffness_matrix(hmat, m, n, h)
    B = constant(compute_interaction_matrix(m, n, h))
    Z = [K -B'
    B spzero(size(B,1))]
    bd = bcnode("lower", m, n, h)
    Z, _ = fem_impose_Dirichlet_boundary_condition(Z, bd, m, n, h)
    T = reshape(repeat(T, 1, 4), (-1,))
    rhs = Ra * compute_fem_source_term(constant(zeros(4*m*n)), T, m, n, h)
    rhs = [rhs;constant(zeros(m*n))]
    rhs = scatter_update(rhs, [bd; bd .+ (m+1)*(n+1)], zeros(2length(bd)))
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
    # η = mantle_viscosity(u, T, m, n, h; kwargs...)
    η = ones(4*m*n)
end

xy = fvm_nodes(m, n, h)
T0 = @. exp(-10*((xy[:,1]-1.25)^2 + (xy[:,2]-0.5)^2))

u_arr = TensorArray(NT+1)
T_arr = TensorArray(NT+1)
p_arr = TensorArray(NT+1)
i = constant(1, dtype = Int32)
u_arr = write(u_arr, 1, zeros(2*(m+1)*(n+1)))
T_arr = write(T_arr, 1, T0)
p_arr = write(p_arr, 1, zeros(m*n))

function condition(i, tas...)
    i<=NT 
end
function body(i, tas...)
    u_arr, T_arr, p_arr = tas
    u, T, p = read(u_arr, i), read(T_arr, i), read(p_arr, i)
    T = solve_heat_eq(u, T)
    η = compute_viscosity_parameter(u, T)
    u, p = solve_stokes(T, η)
    i+1, write(u_arr, i+1, u), write(T_arr, i+1, T), write(p_arr, i+1, p)
end

_, u, T, p = while_loop(condition, body, [i, u_arr, T_arr, p_arr])
u = set_shape(stack(u), (NT+1, 2(m+1)*(n+1)))
T = set_shape(stack(T), (NT+1, m*n))
p = set_shape(stack(p), (NT+1, m*n))

sess = Session(); init(sess)
u_, T_, p_ = run(sess, [u, T, p])


visualize_displacement(u_, m, n, h)
# visualize_scalar_on_fvm_points(T_, m, n, h)
# visualize_scalar_on_fvm_points(p_, m, n, h)