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
dt = 1/NT
Δt = ones(NT)*dt
Δu = 1.0 # fixed Dirichlet boundary condition on the left hand side 
n = 10
m = 10n 
h = 4. /n
left = bcnode("left", m, n, h)
right = bcnode("right", m, n, h)
dof = (m+1)*(n+1)
Force = zeros(NT, dof)


μ = constant(ones(4*m*n))* 100000.
η = constant(ones(4*m*n))  

coef = 2μ*η/(η + μ*dt)
coef = tf.ones_like(coef) * 2.0
mapH = c->begin 
  c * diagm(0=>ones(2))
end
H = map(mapH, coef)

# Precompute the mass matrices, stiffness matrix, and damping matrix 
# H = diagm(0=>ones(2))
M = constant(compute_fem_mass_matrix1(m, n, h))
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
ushift = constant(ub[:,1])
Force = (-K * ub)' 

C = K 
K = spzero((m+1)*(n+1))

strain_rate = zeros((m+1)*(n+1))
for i = 1:m+1
  for j = 1:n+1
    idx = (j-1)*(m+1)+i
    strain_rate[idx] = (m+1-i)/m 
  end
end


# Initial conditions
d0 = zeros(dof)
v0 = zeros(dof)
a0 = zeros(dof)

idof = ones(Bool, dof)
idof[[left;right]] .= false
idof = findall(idof)
# a0 = vector(idof, M[idof, idof]\(Force[1,idof]), dof)
a0 = vector(idof, M[idof, idof]\constant(strain_rate)[idof], dof)





# α scheme time stepping
function solver(A, rhs)
  rhs = scatter_update(rhs, [left;right], zeros(length(left)+length(right)))
  A, _ = fem_impose_Dirichlet_boundary_condition1(A, [left;right], m, n, h)
  A\rhs  
end


function αintegration_visco(M::Union{SparseTensor, SparseMatrixCSC}, 
  C::Union{SparseTensor, SparseMatrixCSC}, 
  K::Union{SparseTensor, SparseMatrixCSC}, 
  Force::Union{Array{Float64}, PyObject}, 
  d0::Union{Array{Float64, 1}, PyObject}, 
  v0::Union{Array{Float64, 1}, PyObject}, 
  a0::Union{Array{Float64, 1}, PyObject}, 
  Δt::Array{Float64}; solve::Union{Missing, Function} = missing)
    n_ = length(Δt)
    ρ = 1.0
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)
    γ = 1/2-αm+αf 
    β = 0.25*(1-αm+αf)^2
    d = length(d0)

    M = isa(M, SparseMatrixCSC) ? constant(M) : M
    C = isa(C, SparseMatrixCSC) ? constant(C) : C
    K = isa(K, SparseMatrixCSC) ? constant(K) : K
    Force, d0, v0, a0, Δt = convert_to_tensor([Force, d0, v0, a0, Δt], [Float64, Float64, Float64, Float64, Float64])

    function equ(dc, vc, ac, σc, εc, dt, Force)
      dn = dc + dt*vc + dt^2/2*(1-2β)*ac 
      vn = vc + dt*((1-γ)*ac)

      df = (1-αf)*dn + αf*dc
      vf = (1-αf)*vn + αf*vc 
      am = αm*ac 

      σn = antiplane_viscosity(-εc/dt, σc, μ, η, dt)
      # σn = -2*εc/dt
      Force_σ = compute_strain_energy_term1(σn, m, n, h) 

      # rhs = Force - (M*am + C*vf + K*df) + K * (dc + ushift)  # - Force_σ
      rhs = - (M*am + C*vf + K*df) 
      A = (1-αm)*M + (1-αf)*C*dt*γ + (1-αf)*K*β*dt^2

      if !ismissing(solve)
        return solve(A, rhs)
      else 
        return A\rhs
      end
    end

    function condition(i, tas...)
      return i<=n_-1
    end
    function body(i, tas...)
      dc_arr, vc_arr, ac_arr, σ_arr, ε_arr = tas
      dc = read(dc_arr, i)
      vc = read(vc_arr, i)
      ac = read(ac_arr, i)
      σc = read(σ_arr, i)
      εc = read(ε_arr, i)
      y = equ(dc, vc, ac, σc, εc, Δt[i], Force[i])
      dn = dc + Δt[i]*vc + Δt[i]^2/2*((1-2β)*ac+2β*y)
      vn = vc + Δt[i]*((1-γ)*ac+γ*y)
      εnew = eval_strain_on_gauss_pts1(dn + ushift, m, n, h)
      σnew = antiplane_viscosity((εnew-εc)/Δt[i], σc, μ, η, Δt[i])
      # σnew = 2*(εnew-εc)/dt
      i+1, write(dc_arr, i+1, dn), write(vc_arr, i+1, vn), 
          write(ac_arr, i+1, y), write(σ_arr, i+1, σnew),
          write(ε_arr, i+1, εnew)
    end

    dM = TensorArray(n_); vM = TensorArray(n_); 
    aM = TensorArray(n_); σM = TensorArray(n_)
    εM = TensorArray(n_)
    dM = write(dM, 1, d0)
    vM = write(vM, 1, v0)
    aM = write(aM, 1, a0)
    σM = write(σM, 1, constant(zeros(4*m*n,2)))
    εM = write(εM, 1, constant(zeros(4*m*n,2)))
    i = constant(1, dtype=Int32)
    _, d, v, a = while_loop(condition, body, [i,dM, vM, aM, σM, εM])
    stack(d), stack(v), stack(a)
end



d, v, a = αintegration_visco(M, C, K, Force, d0, v0, a0, Δt; solve = solver)


# Simulation
sess = Session()
d_, v_, a_ = run(sess, [d, v, a])
# visualize_potential(permutedims(reshape(d_, NT, m+1, n+1), [1,3,2])[:,1:n,1:m], m, n, h)

# close("all")
# for α in [0.25, 0.5, 0.75, 1.0]
#   idx = Int64(round(α/dt))
#   @info idx
#   t = α*tR
#   # κ = 2.
#   # plot((0:m)*h, erfc.((0:m)*h/2/√t/√κ))
#   plot((0:m)*h, d_[idx,  1:m+1], "+--", label="\$t/T=$α\$")
# end
# xlim(0, 5.0)
# legend()


plot(d_[:,1:m+1])
pcolormesh(reshape(d_[end,:], m+1, n+1)')

