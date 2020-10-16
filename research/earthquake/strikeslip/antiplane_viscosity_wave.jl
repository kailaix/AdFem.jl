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


n = 20
NT = 100
ρ = 1.0

m = 5n 
h = 1/n 
Δt = 1.0/NT 

xo = zeros((m+1)*(n+1))
yo = zeros((m+1)*(n+1))
for i = 1:m+1
  for j = 1:n+1
    idx = (j-1)*(m+1)+i 
    xo[idx] = (i-1)*h 
    yo[idx] = (j-1)*h 
  end
end

bdnode = bcnode("left | lower | right", m, n, h)

bdnode = bcnode("all", m, n, h)

μ = 0.5*constant(ones(4*m*n))
η = 1000000. * constant(ones(4*m*n))

coef = 2μ*η/(η + μ*Δt)
mapH = c->begin 
  c * diagm(0=>ones(2))
end
H = map(mapH, coef)

Δt = Δt * ones(NT)

M = constant(compute_fem_mass_matrix1(m, n, h))
K = compute_fem_stiffness_matrix1(H, m, n, h)

db = zeros((m+1)*(n+1))
for j = 1:n+1
  idx = (j-1)*(m+1)+1
  if j<=div(n, 4)
    db[idx] = 1.
  else
    db[idx] = (1-(j-div(n, 4))/(3div(n, 4)))
  end
end

# turn the problem into homogeneous Dirichlet problem
homoF = - K * db
idof = ones(Bool, (m+1)*(n+1))
idof[bdnode] .= false
idof = findall(idof)


d0 = zeros((m+1)*(n+1))
v0 = zeros((m+1)*(n+1))
a0 = vector(idof, M[idof, idof]\((-K*db)[idof]), (m+1)*(n+1))

# analytical solution 
d0 = @. (5-xo)*xo*(1-yo)*yo
v0 = -d0 
a0 = d0

ExtForce = zeros(NT, (m+1)*(n+1))
ts = αscheme_time(Δt)
for i = 1:NT
  t = ts[i]
  f = (x,y)->exp(-t)*(5-x)*x*(1-y)*y - exp(-t)*(-2y*(1-y)-2x*(5-x))
  fval = eval_f_on_gauss_pts(f, m, n, h)
  ExtForce[i,:] = compute_fem_source_term1(fval, m, n, h)
end
ExtForce = constant(ExtForce)

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


  A = (1-αm)*M + (1-αf)*K*β*Δt[1]^2
  A, _ = fem_impose_Dirichlet_boundary_condition1(A, bdnode, m, n, h)


  function equ(dc, vc, ac, dt, εc, σc, i)
    dn = dc + dt*vc + dt^2/2*(1-2β)*ac 
    vn = vc + dt*((1-γ)*ac)

    df = (1-αf)*dn + αf*dc
    vf = (1-αf)*vn + αf*vc 
    am = αm*ac 

    σ_dt = (1-αf)*dt
    Σ = 2* repeat(μ.*η/(η+μ*σ_dt),1,2) .*εc - repeat(η/(η+μ*σ_dt), 1, 2) .* σc
    Force = compute_strain_energy_term1(Σ, m, n, h)

    # rhs = homoF + Force - (M*am + K*df)

    # For wave equation 
    rhs = ExtForce[i] - (M*am + K*df)

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
  # for exact solution 
  # ε0 = eval_strain_on_gauss_pts1(d0, m, n, h)
  # σM = write(σM, 1, 2repeat(μ, 1, 2).*ε0)

  σM = write(σM, 1, zeros(4m*n,2))

  i = constant(1, dtype=Int32)
  _, d, v, a = while_loop(condition, body, [i,dM, vM, aM, σM])
  set_shape(stack(d), (nt+1, length(a0))), set_shape(stack(v), (nt+1, length(a0))), set_shape(stack(a), (nt+1, length(a0)))
end


d, v, a = antiplane_visco_αscheme(M, K, d0, v0, a0, Δt, ρ=ρ)


# function solver(A, rhs)
#   A, _ = fem_impose_Dirichlet_boundary_condition1(A, bdnode, m, n, h)
#   rhs = scatter_update(rhs, bdnode, zeros(length(bdnode)))
#   A\rhs
# end
# d, v, a = αscheme(M, spzero((m+1)*(n+1), (m+1)*(n+1)), K, ExtForce, d0, v0, a0, Δt; solve = solver, ρ=ρ)


sess = Session(); init(sess)
d_, v_, a_ = run(sess, [d, v, a])


# pcolormesh(reshape(d_[2,:]+db, m+1, n+1))
# colorbar()

close("all")
for (k,tid) in enumerate(LinRange{Int64}(1, NT+1, 5))
  t = (tid-1)*Δt[1]
  dd = @. exp(-t)*(5-xo)*xo*(1-yo)*yo
  plot(dd[div(n,2)*(m+1) .+ (1:m+1)], "C$k-", label="$tid")
  plot(d_[tid, :][div(n,2)*(m+1) .+ (1:m+1)],"C$k--o", label="$tid", markersize=3)
end
legend()


# close("all")
# for tid in LinRange{Int64}(1, NT+1, 5)
#   plot((a_[tid, :])[1:m+1], label="$tid")
# end
# legend()


