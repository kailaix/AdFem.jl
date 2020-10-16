using Revise
using AdFem
using PyPlot

m = 30
n = 30
h = 1/n 

xy = fem_nodes(m, n, h) 
x, y = xy[:,1], xy[:,2]
u0 = [@. (2(x^3/3-x^2/2)); @. (2(y^3/3 - y^2/2))]

xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @. x*(1-x)*y*(1-y)

ρ = dt = 1
function step2(u0)
    B = compute_interaction_matrix(m, n, h)
    bc = bcedge("all", m, n, h)
    A, _ = compute_fvm_tpfa_matrix(ones(m*n), bc, zeros(size(bc,1)), m, n, h)
    rhs = ρ / dt * B * u0
    sol = A\rhs
    return sol
end
bc = bcedge("all", m, n, h)

# There are three ways to compute 
# int div u dx
# on each element

# rhs = compute_fvm_mechanics_term(u0, m, n, h)
B = compute_interaction_matrix(m, n, h)
rhs = B * u0
# rhs = h^2 * @. -2x*(1-x)-2y*(1-y)

A, _ = compute_fvm_tpfa_matrix(ones(m*n), bc, zeros(size(bc,1)), m, n, h)
sol = A\rhs

sol = step2(u0)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fvm_points(sol, m, n, h)
subplot(122)
visualize_scalar_on_fvm_points(p0, m, n, h)
