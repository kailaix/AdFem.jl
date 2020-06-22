using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
np = pyimport("numpy")

# mode = "data" generate data 
# mode != "data" train 
mode = "train"


β = 1/4; γ = 1/2
a = b = 0.1
m = 40
n = 20
h = 0.01
NT = 50
Δt = 5/NT 
bdedge = []
for j = 1:n 
  push!(bdedge, [(j-1)*(m+1)+m+1 j*(m+1)+m+1])
end
bdedge = vcat(bdedge...)

bdnode = Int64[]
for j = 1:n+1
  push!(bdnode, (j-1)*(m+1)+1)
end

# λ = Variable(1.0)
# μ = Variable(1.0)
# invη = Variable(1.0)

λ = constant(2.0)
μ = constant(0.5)
if mode=="data"
  global invη = placeholder(12.0)
else
  global invη = Variable(1.0)
end

G = tensor([1/Δt+μ*invη -μ/3*invη 0.0
  -μ/3*invη 1/Δt+μ*invη-μ/3*invη 0.0
  0.0 0.0 1/Δt+μ*invη])
S = tensor([2μ/Δt+λ/Δt λ/Δt 0.0
    λ/Δt 2μ/Δt+λ/Δt 0.0
    0.0 0.0 μ/Δt])

invG = inv(G)
# invG = Variable([1.0 0.0 0.0
#                 0.0 1.0 0.0
#                 0.0 0.0 1.0])
H = invG*S


M = compute_fem_mass_matrix1(m, n, h)
Zero = spzeros((m+1)*(n+1), (m+1)*(n+1))
M = SparseTensor([M Zero;Zero M])

K = compute_fem_stiffness_matrix(H, m, n, h)
C = a*M + b*K # damping matrix 

L = M + γ*Δt*C + β*Δt^2*K
L, Lbd = fem_impose_Dirichlet_boundary_condition(L, bdnode, m, n, h)
# error()


a = TensorArray(NT+1); a = write(a, 1, zeros(2(m+1)*(n+1))|>constant)
v = TensorArray(NT+1); v = write(v, 1, zeros(2(m+1)*(n+1))|>constant)
d = TensorArray(NT+1); d = write(d, 1, zeros(2(m+1)*(n+1))|>constant)
U = TensorArray(NT+1); U = write(U, 1, zeros(2(m+1)*(n+1))|>constant)
Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(4*m*n, 3)|>constant)
Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1,zeros(4*m*n, 3)|>constant)


Forces = zeros(NT, 2(m+1)*(n+1))
for i = 1:NT
  T = eval_f_on_boundary_edge((x,y)->0.1, bdedge, m, n, h)
  T = [T zeros(length(T))]
  rhs = compute_fem_traction_term(T, bdedge, m, n, h)

  if i*Δt>3.0
    rhs = zero(rhs)
  end
  Forces[i, :] = rhs
end
Forces = constant(Forces)

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

  F = compute_strain_energy_term(Sigma*invG/Δt, m, n, h) - K * U
  rhs = Forces[i] - F
  td = d + Δt*v + Δt^2/2*(1-2β)*a 
  tv = v + (1-γ)*Δt*a 
  rhs = rhs - C*tv - K*td
  
  # rhs[[bdnode; bdnode.+(m+1)*(n+1)]] .= 0.0

  rhs = scatter_update(rhs, constant([bdnode; bdnode.+(m+1)*(n+1)]), constant(zeros(2*length(bdnode))))

  a = L\rhs # bottleneck  
  d = td + β*Δt^2*a 
  v = tv + γ*Δt*a 
  U_new = d
  Varepsilon_new = eval_strain_on_gauss_pts(U_new, m, n, h)
  Sigma_new = Sigma*invG/Δt +  (Varepsilon_new-Varepsilon)*(invG*S)

  i+1, write(a_, i+1, a), write(v_, i+1, v), write(d_, i+1, d), write(U_, i+1, U_new),
        write(Sigma_, i+1, Sigma_new), write(Varepsilon_, i+1, Varepsilon_new)
end


i = constant(1, dtype=Int32)
# error()
_, _, _, _, u, sigma, varepsilon = while_loop(condition, body, 
                  [i, a, v, d, U, Sigma, Varepsilon])

U = stack(u)
Sigma = stack(sigma)
Varepsilon = stack(varepsilon)

if mode!="data"
  global Uval = matread("U.mat")["U"]
  U.set_shape((NT+1, size(U, 2)))
  idx = 1:m+1
  global loss = sum((U[:, idx] - Uval[:, idx])^2)
end

# opt = AdamOptimizer(1.0).minimize(loss)
sess = Session(); init(sess)

if mode=="data"
  matwrite("U.mat", Dict("U"=>run(sess, U)))
  error()
end


@info run(sess, loss)
v_ = []
i_ = []
l_ = []
cb = (v, i, l)->begin
  println("[$i] loss = $l")
  println(v)
  push!(v_, [x for x in v])
  push!(i_, i)
  push!(l_, l)
end
loss_ = BFGS!(sess, loss, vars=[λ, μ, invη], callback=cb)
# matwrite("R.mat", Dict("V"=>v_, "L"=>l_))

# for i = 1:1000
#   _, l, invη_ = run(sess, [opt, loss, invη])
#   @show i, l, invη_
# end



# ηs = 11:0.1:13
# losses = []
# for η in ηs
#    push!(losses,run(sess, loss, invη=>η))
# end
# plot(ηs, losses)


# Uval, Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])
# Uval[idx]



# visualize_displacement(U, m, n, h; name = "_eta$η", xlim_=[-0.01,0.5], ylim_=[-0.05,0.22])
# # visualize_displacement(U, m, n, h;  name = "_viscoelasticity")
# # visualize_stress(H, U, m, n, h;  name = "_viscoelasticity")

# close("all")
# figure(figsize=(15,5))
# subplot(1,3,1)
# idx = div(n,2)*(m+1) + m+1
# plot((0:NT)*Δt, Uval[:, idx])
# xlabel("time")
# ylabel("\$u_x\$")

# subplot(1,3,2)
# idx = 4*(div(n,2)*m + m)
# plot((0:NT)*Δt, Sigmaval[:,idx,1])
# xlabel("time")
# ylabel("\$\\sigma_{xx}\$")

# subplot(1,3,3)
# idx = 4*(div(n,2)*m + m)
# plot((0:NT)*Δt, Varepsilonval[:,idx,1])
# xlabel("time")
# ylabel("\$\\varepsilon_{xx}\$")
# savefig("visco_eta$η.png")
