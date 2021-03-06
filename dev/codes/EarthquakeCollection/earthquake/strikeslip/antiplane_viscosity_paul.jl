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

# simulation parameter setup
n = 20
NT = 400
ρ = 0.1 # design variable in α-schemes
m = 5n 
h = 1/n 
Δt = 10000. /NT 

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
bdnode = bcnode("left  | lower | right", m, n, h)

# viscoelasticity η and shear modulus μ

ηf = (x,y)->begin 
  if y<=0.25
    return 10000.
  else 
    return 1.
  end
end

η = constant(eval_f_on_gauss_pts(ηf, m, n, h))
μ = 0.001 * constant(ones(4m*n))




# linear elasticity matrix 
coef = 2μ*η/(η + μ*Δt)
mapH = c->begin 
  c * diagm(0=>ones(2))
end
H = map(mapH, coef)

Δt = Δt * ones(NT)

# generalized mass matrix and stiffness matrix 
M = 100. * constant(compute_fem_mass_matrix1(m, n, h))
K = compute_fem_stiffness_matrix1(H, m, n, h)
C = spzero((m+1)*(n+1))

C = 0.1 * M + 0.1 * K 

# fixed displacement 
db = zeros((m+1)*(n+1))
for j = 1:n+1
  idx = (j-1)*(m+1)+1
  if j<=div(n, 4)
    db[idx] = 1.
  else
    db[idx] = (1-(j-div(n, 4))/(3div(n, 4)))
  end
  # y = (j-1)*h
  # db[idx] = y * (1-y) * 0.0
end

# cast the problem into homogeneous Dirichlet problem
idof = ones(Bool, (m+1)*(n+1))
idof[bdnode] .= false
idof = findall(idof)

d0 = db
v0 = zeros((m+1)*(n+1))
# v0[bcnode("left", m, n, h)] .= 1.0

a0 = vector(idof, M[idof, idof]\((-K*db)[idof]), (m+1)*(n+1))


function solver(A, rhs)
  A, _ = fem_impose_Dirichlet_boundary_condition1(A, bdnode, m, n, h)
  rhs = scatter_update(rhs, bdnode, zeros(length(bdnode)))
  A\rhs
end


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

    rhs = - (M*am + C*vf + K*df) + Force
    
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


    y = equ(dc, vc, ac, Δt[i],εc, σc, i)
    dn = dc + Δt[i]*vc + Δt[i]^2/2*((1-2β)*ac+2β*y)
    vn = vc + Δt[i]*((1-γ)*ac+γ*y)

    εn = eval_strain_on_gauss_pts1(dn, m, n, h)
    σn = 2* repeat(μ.*η/(η+μ*Δt[i]),1,2) .* (εn - εc) + repeat(η/(η+μ*Δt[i]), 1, 2) .* σc

    i+1, write(dc_arr, i+1, dn), write(vc_arr, i+1, vn), write(ac_arr, i+1, y), write(σc_arr, i+1, σn)
  end

  dM = TensorArray(nt+1); vM = TensorArray(nt+1); aM = TensorArray(nt+1); σM = TensorArray(nt+1);
  dM = write(dM, 1, d0)
  vM = write(vM, 1, v0)
  aM = write(aM, 1, a0)

  ε0 = eval_strain_on_gauss_pts1(d0, m, n, h)
  σ0 = batch_matmul(H, ε0)
  σM = write(σM, 1, σ0)

  i = constant(1, dtype=Int32)
  _, d, v, a = while_loop(condition, body, [i,dM, vM, aM, σM])
  set_shape(stack(d), (nt+1, length(a0))), set_shape(stack(v), (nt+1, length(a0))), set_shape(stack(a), (nt+1, length(a0)))
end

# d, v, a = αscheme(M, C, K, zeros(NT, (m+1)*(n+1)), d0, v0, a0, Δt; solve = solver)

d, v, a = antiplane_visco_αscheme(M, K, d0, v0, a0, Δt, ρ=ρ)


sess = Session(); init(sess)
d_, v_, a_ = run(sess, [d, v, a])


# close("all")
# for (k,tid) in enumerate(LinRange{Int64}(1, NT+1, 5))
#   t = (tid-1)*Δt[1]
#   plot((d_[tid, :]+db)[(1:m+1)],"C$k-", label="$t", markersize=3)
# end
# legend()


# close("all")
# for tid in LinRange{Int64}(1, NT+1, 5)
#   plot((a_[tid, :])[1:m+1], label="$tid")
# end
# legend()


# plot(d_[end,1:m+1])
# plot(d_[div(NT+1,2),1:m+1])
# plot(d_[div(NT+1,3),1:m+1])


# pcolormesh(reshape(d_[end,:], m+1, n+1)')
figure()
pl, = plot([], [], "o-", markersize = 3)
t = title("0")
xi = (0:m)*h 
xlim(-h, (m+1)*h)
xlabel("Distance")
ylim(-0.1, 1.1)
ylabel("Displacement")
tight_layout()
function update(i)
  pl.set_data(xi[:], d_[i,1:m+1])
  t.set_text("time = $(i*Δt[1])")
end
p = animate(update, [1:1:20;25:5:NT+1])
saveanim(p, "displacement.gif")

figure()
pl, = plot([], [], "o-", markersize = 3)
t = title("time = 0")
xi = (0:m)*h 
xlim(-h, (m+1)*h)
xlabel("Distance")
ylim(-0.0001, 0.0005)
ylabel("Velocity")
tight_layout()
function update(i)
  pl.set_data(xi[:], v_[i,1:m+1])
  t.set_text("time = $(i*Δt[1])")
end
p = animate(update, [1:1:20;25:5:NT+1])
saveanim(p, "velocity.gif")

figure()
pl, = plot([], [], "o-", markersize = 3)
t = title("time = 0")
xi = (0:m)*h 
xlim(-h, (m+1)*h)
xlabel("Distance")
ylim(-0.001, 0.001)
ylabel("Strain Rate")
tight_layout()
function update(i)
  pl.set_data(xi[1:end-1], (v_[i,2:m+1]-v_[i,1:m])/h)
  t.set_text("time = $(i*Δt[1])")
end
p = animate(update, [1:1:20;25:5:NT+1])
saveanim(p, "strain_rate.gif")
