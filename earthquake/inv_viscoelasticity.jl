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

h1 = 1.
h2 = 3.
μ1 = 1.
μ2 = 1.
η1 = 0.
η2 = 1.
Δu = 1.
κ = h2*h1*μ1/η2
tR = h1 * η2 / (h2 * μ1)
x_ = range(0, 5, length=100)
t_ = range(0, 5, length=100)
x = x_ * h1
t = t_ * tR 

# # u = Δu * (1 - erf(x/(2*κ*t)))
# # v = @. 1/t_ * x_ / (2*sqrt(π * t_)) * exp(-x_^2 / (4*t_))

# ## displacement
# figure()
# for t_ in [0.25, 0.5, 1, 2]
# u_ = @. 1 - erf(x_/(2*sqrt(t_)))
# plot(x_, u_, label="t/tR=$(t_)")
# end
# legend()

# ## velocity
# figure()
# for t_ in [0.25, 0.5, 1, 2]
# v = @. 1/t_ * x_ / (2*sqrt(π * t_)) * exp(-x_^2 / (4*t_))
# plot(x_, v, label="t/tR=$(t_)")
# end
# legend()

# ## strain rate
# figure()
# for t_ in [0.25, 0.5, 1, 2]
# dγdt = @. 1/t_ / ( 2 * sqrt(π * t_) ) * ( 1 - x_^2 / (2*t_) ) * exp( -x_^2 / (4*t_) )
# plot(x_, dγdt, label="t/tR=$(t_)")
# end
# legend()

# ## strain rate
# figure()
# x_ = 0
# t_ = range(0, 5, length=100)
# dγdt = @. 1/t_ / ( 2 * sqrt(π * t_) ) * ( 1 - x_^2 / (2*t_) ) * exp( -x_^2 / (4*t_) )
# plot(t_, dγdt)
# legend()



h = 0.05
NT = 100


stepsize = 1
if length(ARGS)==1
  global stepsize = parse(Int64, ARGS[1])
end
@info stepsize

mode = "data"
## alpha-scheme
β = 1/4
γ = 1/2
a = b = 0.1

H = 1.0
H2 = 3.0
Depth = H + H2 
Width = 5H 
n = Int64(round(Depth/h))
m = Int64(round(Width/h))

T = 5
Δt = T/NT


bdright = bcnode("right", m, n, h)
bdleft = bcnode("left", m, n, h)

μ = constant(ones(4*m*n))
η = constant(ones(4*m*n))

coef = 2μ*η/(η + μ*Δt)
mapH = c->begin 
  c * diagm(0=>ones(2))
end
H = map(mapH, coef)

M = constant(compute_fem_mass_matrix1(m, n, h))
K = compute_fem_stiffness_matrix1(H, m, n, h)
C = a*M + b*K # damping matrix 
L = M + γ*Δt*C + β*Δt^2*K
L, Lbd = fem_impose_Dirichlet_boundary_condition1(L, [bdleft; bdright], m, n, h)

ubd = zeros(length(bdleft)+length(bdright))
ubd[1:length(bdleft)] .= Δu
ubd = constant(ubd)

a = TensorArray(NT+1); a = write(a, 1, zeros((m+1)*(n+1))|>constant)
v = TensorArray(NT+1); v = write(v, 1, zeros((m+1)*(n+1))|>constant)
d = TensorArray(NT+1); d = write(d, 1, zeros((m+1)*(n+1))|>constant)
U = TensorArray(NT+1); U = write(U, 1, zeros((m+1)*(n+1))|>constant)
Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(4*m*n, 2)|>constant)
Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1,zeros(4*m*n, 2)|>constant)



function condition(i, tas...)
  i <= NT
end

function body(i, tas...)
  a_, v_, d_, U_, Sigma_, Varepsilon_ = tas
  a = read(a_, i)
  v = read(v_, i)
  d = read(d_, i)
  U = read(U_, i)
  Sigma = read(Sigma_, i)
  Varepsilon = read(Varepsilon_, i)

  σn = antiplane_viscosity(-Varepsilon/Δt, Sigma, μ, η, Δt)
  F = compute_strain_energy_term1(σn, m, n, h) - K * U
  rhs = - Δt^2 * F

  td = d + Δt*v + Δt^2/2*(1-2β)*a 
  tv = v + (1-γ)*Δt*a 
  rhs = rhs - C*tv - K*td
  rhs = scatter_update(rhs, constant([bdleft;bdright]), zeros(length(bdleft)+length(bdright)))


  ## alpha-scheme
  a = L\rhs # bottleneck  
  d = td + β*Δt^2*a 
  v = tv + γ*Δt*a 
  d = scatter_update(d, constant([bdleft;bdright]), ubd)
  U_new = d

  Varepsilon_new = eval_strain_on_gauss_pts1(U_new, m, n, h)

  Sigma_new = antiplane_viscosity((Varepsilon_new-Varepsilon)/Δt , Sigma, μ, η, Δt)

  i+1, write(a_, i+1, a), write(v_, i+1, v), write(d_, i+1, d), write(U_, i+1, U_new),
        write(Sigma_, i+1, Sigma_new), write(Varepsilon_, i+1, Varepsilon_new)
end


i = constant(1, dtype=Int32)
_, _, _, _, u, sigma, varepsilon = while_loop(condition, body, 
                  [i, a, v, d, U, Sigma, Varepsilon])

U = stack(u)
Sigma = stack(sigma)
Varepsilon = stack(varepsilon)

sess = Session(); init(sess)

Uval,Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])
close("all")
for α in [0.25, 0.5, 1.0, 2.0]
  idx = Int64(round((tR*α/T)*NT)) + 1
  plot((0:m)*h, Uval[idx,  1:m+1], label="\$t/t_R=$α\$")
end
