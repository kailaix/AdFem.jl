using Revise
using AdFem
using PyPlot
using ADCME

m = 100
n = 100
h = 1/n 
Δt = 0.1
ρ = 1


function step3(u0, dp)
    A = constant(compute_fem_mass_matrix(m, n, h))
    bd = bcnode("all", m, n, h)

    grad_dp_fem = - compute_interaction_term(dp, m, n, h)
    b = A * u0 - Δt/ρ * grad_dp_fem

    A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h) # 2D boundary condition
    bd = [bd; (m+1)*(n+1) .+ bd]
    b = scatter_update(b, bd, zeros(length(bd)))
    sol = A\b
    return sol
end


xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
dp = @. x^2*(1-x)^2*y^2*(1 -y)^2


xy = fem_nodes(m, n, h) 
x, y = xy[:,1], xy[:,2]
u0 = [@. (x*(1-x)*y*(1-y)); @. (x*(1-x)*y*(1-y))]

grad_dp_x = x.^2 .*y.^2 .*(1 .- y).^2 .*(2*x .- 2) + 2*x.*y.^2 .*(1 .- x).^2 .*(1 .- y).^2
grad_dp_y = x.^2 .*y.^2 .*(1 .- x).^2 .*(2*y .- 2) + 2*x.^2 .*y.*(1 .- x).^2 .*(1 .- y).^2

# u1 = [@. (x*(1-x)*y*(1-y)- Δt * (1-2x)*y*(1-y) ); @. (x*(1-x)*y*(1-y) - Δt * (1-2y)*x*(1-x) )]
u1_x = u0[1:(m+1)*(n+1)]- Δt / ρ * grad_dp_x
u1_y = u0[(m+1)*(n+1)+1:end] - Δt / ρ *grad_dp_y

# A = constant(compute_fem_mass_matrix(m, n, h))
# bd = bcnode("all", m, n, h)

# grad_dp_fem = - compute_interaction_term(dp, m, n, h)
# b = A * u0 - Δt/ρ * grad_dp_fem

# A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h) # 2D boundary condition
# bd = [bd; (m+1)*(n+1) .+ bd]
# b = scatter_update(b, bd, zeros(length(bd)))

# sol = A\b

sol = step3(u0, dp)

sess = Session(); init(sess)
sol = run(sess, sol)
# u1 = run(sess, u1)

# exact = u1(x, y)


figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(sol[1:(m+1)*(n+1)]-u1_x, m, n, h)
subplot(122)
visualize_scalar_on_fem_points(u1_x, m, n, h)



figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(sol[(m+1)*(n+1)+1:end]-u1_y, m, n, h)
subplot(122)
visualize_scalar_on_fem_points(u1_y, m, n, h)