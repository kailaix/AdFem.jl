using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
using ADCMEKit
np = pyimport("numpy")
using PyPlot
using SpecialFunctions

# simulation parameter setup
n = 12
NT = 200
ρ = 0.9 # design variable in α-schemes

m = 5n 
h = 1/n 
Δt = 1/NT 

# coordinates
xo = zeros((m+1)*(n+1))
yo = zeros((m+1)*(n+1))
for i = 1:m+1
  for j = 1:n+1
    idx = (j-1)*(m+1)+i 
    xo[idx] = (i-1)*h 
    yo[idx] = (j-1)*h 
  end
end

# Dirichlet boundary condition on three sides, and Neumann boundary condition (traction-free) on the top
bdnode = bcnode("left | lower | right", m, n, h)

# viscoelasticity η and shear modulus μ

ηf = (x,y)->begin 
  if y<=0.25
    return 100000.
  else 
    return 100000.
  end
end

η = constant(eval_f_on_gauss_pts(ηf, m, n, h))
μ = constant(ones(4m*n))




# linear elasticity matrix 
coef = 2μ*η/(η + μ*Δt)
mapH = c->begin 
  c * diagm(0=>ones(2))
end
H = map(mapH, coef)

Δt = Δt * ones(NT)

# generalized mass matrix and stiffness matrix 
M = constant(compute_fem_mass_matrix1(m, n, h))
K = compute_fem_stiffness_matrix1(H, m, n, h)


# fixed displacement 
db = zeros((m+1)*(n+1))
for j = 1:n+1
  idx = (j-1)*(m+1)+1
  if j<=div(n, 4)
    db[idx] = 1.
  else
    db[idx] = (1-(j-div(n, 4))/(3div(n, 4)))
  end
end

bdval = zeros(length(bdnode))
for i = 1:length(bdnode)
  bdval[i] = db[bdnode[i]]
end

# cast the problem into homogeneous Dirichlet problem
idof = ones(Bool, (m+1)*(n+1))
idof[bdnode] .= false
idof = findall(idof)

d0 = db
v0 = zeros((m+1)*(n+1))
a0 = vector(idof, M[idof, idof]\((-K*db)[idof]), (m+1)*(n+1))

function antiplane_visco_αscheme(M::Union{SparseTensor, SparseMatrixCSC}, 
  K::Union{SparseTensor, SparseMatrixCSC}, 
  d0::Union{Array{Float64, 1}, PyObject}, 
  v0::Union{Array{Float64, 1}, PyObject}, 
  a0::Union{Array{Float64, 1}, PyObject}, 
  Δt::Array{Float64}; 
  ρ::Float64 = 1.0)
  nt = length(Δt)
  αm = (2ρ-1)/(ρ+1)
  αf = ρ/(1+ρ)
  γ = 1/2-αm+αf 
  β = 0.25*(1-αm+αf)^2
  d = length(d0)

  M = isa(M, SparseMatrixCSC) ? constant(M) : M
  K = isa(K, SparseMatrixCSC) ? constant(K) : K
  d0, v0, a0, Δt = convert_to_tensor([d0, v0, a0, Δt], [Float64, Float64, Float64, Float64])


  # A = (1-αm)*M + (1-αf)*K*β*Δt[1]^2
  # A, Abd = fem_impose_Dirichlet_boundary_condition1(A, bdnode, m, n, h)

  
  M, Mbd = fem_impose_Dirichlet_boundary_condition1(M, bdnode, m, n, h)
  K, Kbd = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
  A = (1-αm)*M + (1-αf)*K*β*Δt[1]^2



  function equ(dc, vc, ac, dt, εc, σc, i)
    dn = dc + dt*vc + dt^2/2*(1-2β)*ac 
    vn = vc + dt*((1-γ)*ac)

    df = (1-αf)*dn + αf*dc
    vf = (1-αf)*vn + αf*vc 
    am = αm*ac 

    σ_dt = (1-αf)*dt
    Σ = 2* repeat(μ.*η/(η+μ*σ_dt),1,2) .*εc - repeat(η/(η+μ*σ_dt), 1, 2) .* σc
    Force = compute_strain_energy_term1(Σ, m, n, h)

    rhs = - (M*am + K*df) + Force
    rhs -= Kbd*bdval
    
    rhs = scatter_update(rhs, bdnode, zeros(length(bdnode)))
    A\rhs 
    
  end

  function condition(i, tas...)
    return i<=nt
  end
  function body(i, tas...)
    dc_arr, vc_arr, ac_arr, σc_arr = tas
    dc = read(dc_arr, i)
    vc = read(vc_arr, i)
    ac = read(ac_arr, i)
    σc = read(σc_arr, i)
    εc = eval_strain_on_gauss_pts1(dc, m, n, h)

    op = tf.print(i, norm(σc - εc))
    i = bind(i, op)

    y = equ(dc, vc, ac, Δt[i],εc, σc, i)
    dn = dc + Δt[i]*vc + Δt[i]^2/2*((1-2β)*ac+2β*y)
    vn = vc + Δt[i]*((1-γ)*ac+γ*y)

    εn = eval_strain_on_gauss_pts1(dn, m, n, h)
    σn = 2* repeat(μ.*η/(η+μ*Δt[i]),1,2) .* (εn - εc) + repeat(η/(η+μ*Δt[i]), 1, 2) .* σc

    op = tf.print(i, norm(σn - εn))
    i = bind(i, op)

    i+1, write(dc_arr, i+1, dn), write(vc_arr, i+1, vn), write(ac_arr, i+1, y), write(σc_arr, i+1, σn)
  end

  dM = TensorArray(nt+1); vM = TensorArray(nt+1); aM = TensorArray(nt+1); σM = TensorArray(nt+1);
  dM = write(dM, 1, d0)
  vM = write(vM, 1, v0)
  aM = write(aM, 1, a0)
  σM = write(σM, 1, zeros(4m*n,2))

  i = constant(1, dtype=Int32)
  _, d, v, a = while_loop(condition, body, [i,dM, vM, aM, σM])
  set_shape(stack(d), (nt+1, length(a0))), set_shape(stack(v), (nt+1, length(a0))), set_shape(stack(a), (nt+1, length(a0)))
end


d, v, a = antiplane_visco_αscheme(M, K, d0, v0, a0, Δt, ρ=ρ)


sess = Session(); init(sess)
d_, v_, a_ = run(sess, [d, v, a])


close("all")
for (k,tid) in enumerate(LinRange{Int64}(1, NT+1, 5))
  t = (tid-1)*Δt[1]
  plot((d_[tid, :]+db)[div(n,2)*(m+1) .+ (1:m+1)],"C$k-", label="$t", markersize=3)
end
legend()


# close("all")
# for tid in LinRange{Int64}(1, NT+1, 5)
#   plot((a_[tid, :])[1:m+1], label="$tid")
# end
# legend()


