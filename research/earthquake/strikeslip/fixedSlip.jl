using Revise
using AdFem
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
using ADCMEKit
np = pyimport("numpy")
using PyPlot
using SpecialFunctions

# Parameters setup 
NT = 100
Δt = ones(NT)*0.0666
Δu = 1.0 # fixed Dirichlet boundary condition on the left hand side 
n = 10
m = 5n 
h = 0.4
left = bcnode("left", m, n, h)
right = bcnode("right", m, n, h)
dof = (m+1)*(n+1)
Force = zeros(NT, dof)

# Precompute the mass matrices, stiffness matrix, and damping matrix 
H = diagm(0=>ones(2))
M = compute_fem_mass_matrix1(m, n, h)
K = compute_fem_stiffness_matrix1(H, m, n, h)
C = 0.0 * M + 0.0 * K

# Compute the force term 
αt = αscheme_time(Δt)
ub = zeros(dof, NT)
for i = 1:NT 
  ub[left, i] .= Δu
  ub[right, i] .= 0.0
  # Force at αt[i]
end
Force = (-K * ub)'|>Array

# Initial conditions
d0 = zeros(dof)
v0 = zeros(dof)
a0 = zeros(dof)

idof = ones(Bool, dof)
idof[[left;right]] .= false

a0[idof] = M[idof, idof]\(Force[1,idof])

# α scheme time stepping
function solver(A, rhs)
  rhs = scatter_update(rhs, [left;right], zeros(length(left)+length(right)))
  A, _ = fem_impose_Dirichlet_boundary_condition1(A, [left;right], m, n, h)
  A\rhs  
end

d, v, a = αscheme(M, C, K, Force, d0, v0, a0, Δt; solve = solver)

# Simulation
sess = Session()
d_, v_, a_ = run(sess, [d, v, a])

d_[:, left] .+= Δu
# visualize_potential(permutedims(reshape(d_, NT, m+1, n+1), [1,3,2])[:,1:n,1:m], m, n, h)

close("all")
for (k,tid) in enumerate(LinRange{Int64}(1, NT+1, 5))
  t = (tid-1)*Δt[1]
  plot((d_[tid, :])[(1:m+1)],"C$k-", label="$t", markersize=3)
end
legend()
