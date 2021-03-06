# implicit scheme for advection-diffusion

using Revise
using AdFem
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


u = 0.5*[ones(m*n);zeros(m*n)]

up_and_down = bcedge("upper|lower", m, n, h)

M = compute_fvm_mass_matrix(m, n, h)
K, rhs1 = compute_fvm_advection_matrix(u, up_and_down, zeros(size(up_and_down, 1)), m, n, h)
S, rhs2 = compute_fvm_tpfa_matrix(missing, up_and_down, zeros(size(up_and_down, 1)), m, n, h)

A = M/Δt + K - 0.01*S 
A = factorize(A)

U = zeros(m*n, NT+1)
xy = fvm_nodes(m, n, h)
u0 = @. exp( - 10 * ((xy[:,1]-1.0)^2 + (xy[:,2]-0.5)^2))

function condition(i, args...)
    i <= NT
end

function body(i, u_arr)
    u = read(u_arr, i)
    u_arr = write(u_arr, i+1, A\(M*u/Δt - rhs1 + rhs2))
    return i+1, u_arr
end

i = constant(1, dtype = Int32)
u_arr = TensorArray(NT+1)
u_arr = write(u_arr, 1, u0)
_, u = while_loop(condition, body, [i, u_arr])
u = set_shape(stack(u), (NT+1, m*n))


sess = Session(); init(sess)
U = run(sess, u)
Z = zeros(NT+1, n, m)
for i = 1:NT+1
    Z[i,:,:] = reshape(U[i,:], m, n)'
end
p = visualize_scalar_on_fvm_points(Z, m, n, h)
saveanim(p, "advec.gif")

