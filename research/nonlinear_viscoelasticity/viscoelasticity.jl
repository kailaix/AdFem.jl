using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
np = pyimport("numpy")


mode = "data"
sigmamax = 0.1
# pl = placeholder([5.0;1000.0])

pl = Variable([1.0;100.0])


n = 10
m = 2n 
h = 0.01
NT = 20
Δt = 2/NT

bdedge = bcedge("right", m, n, h)
bdnode = bcnode("lower", m, n, h)

λ = constant(2.0)
μ = constant(0.2)


M = compute_fem_mass_matrix1(m, n, h)
Zero = spzeros((m+1)*(n+1), (m+1)*(n+1))
M = SparseTensor([M Zero;Zero M])

## alpha-scheme
β = 1/4; γ = 1/2

function eta_fun(σ)
  return constant(10*ones(4*m*n)) + pl[1]/(1+pl[2]*sum(σ^2, dims=2))
end


# invη is a 4*m*n array 
function make_matrix(invη)
  a = b = 0.1
  fn_G = invη->begin 
    G = tensor([1/Δt+μ*invη -μ/3*invη 0.0
      -μ/3*invη 1/Δt+μ*invη-μ/3*invη 0.0
      0.0 0.0 1/Δt+μ*invη])
    invG = inv(G)
  end
  invG = map(fn_G, invη)
  S = tensor([2μ/Δt+λ/Δt λ/Δt 0.0
      λ/Δt 2μ/Δt+λ/Δt 0.0
      0.0 0.0 μ/Δt])

  H = invG*S
  K = compute_fem_stiffness_matrix(H, m, n, h)
  C = a*M + b*K # damping matrix 
  L = M + γ*Δt*C + β*Δt^2*K
  # L = pl[1]*L
  L, Lbd = fem_impose_Dirichlet_boundary_condition_experimental(L, bdnode, m, n, h)

  return C, K, L, S, invG
end


a = TensorArray(NT+1); a = write(a, 1, zeros(2(m+1)*(n+1))|>constant)
v = TensorArray(NT+1); v = write(v, 1, zeros(2(m+1)*(n+1))|>constant)
d = TensorArray(NT+1); d = write(d, 1, zeros(2(m+1)*(n+1))|>constant)
U = TensorArray(NT+1); U = write(U, 1, zeros(2(m+1)*(n+1))|>constant)
Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(4*m*n, 3)|>constant)
Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1,zeros(4*m*n, 3)|>constant)


Forces = zeros(NT, 2(m+1)*(n+1))
for i = 1:NT
  T = eval_f_on_boundary_edge((x,y)->0.1, bdedge, m, n, h)

  T = [-T T]
  rhs = compute_fem_traction_term(T, bdedge, m, n, h)

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

  invη = eta_fun(Sigma)
  C, K, L, S, invG = make_matrix(invη)

  res = batch_matmul(invG/Δt, Sigma)
  F = compute_strain_energy_term(res, m, n, h) - K * U
  rhs = Forces[i] - Δt^2 * F

  td = d + Δt*v + Δt^2/2*(1-2β)*a 
  tv = v + (1-γ)*Δt*a 
  rhs = rhs - C*tv - K*td
  rhs = scatter_update(rhs, constant([bdnode; bdnode.+(m+1)*(n+1)]), constant(zeros(2*length(bdnode))))


  ## alpha-scheme
  a = L\rhs # bottleneck  
  d = td + β*Δt^2*a 
  v = tv + γ*Δt*a 
  U_new = d

  Varepsilon_new = eval_strain_on_gauss_pts(U_new, m, n, h)

  res2 = batch_matmul(invG * S, Varepsilon_new-Varepsilon)
  Sigma_new = res +  res2

  i+1, write(a_, i+1, a), write(v_, i+1, v), write(d_, i+1, d), write(U_, i+1, U_new),
        write(Sigma_, i+1, Sigma_new), write(Varepsilon_, i+1, Varepsilon_new)
end


i = constant(1, dtype=Int32)
_, _, _, _, u, sigma, varepsilon = while_loop(condition, body, 
                  [i, a, v, d, U, Sigma, Varepsilon])

U = stack(u)
Sigma = stack(sigma)
Varepsilon = stack(varepsilon)

# if mode!="data"
#   data = matread("viscoelasticity.mat")
#   global Uval,Sigmaval, Varepsilonval = data["U"], data["Sigma"], data["Varepsilon"]
#   U.set_shape((NT+1, size(U, 2)))
#   idx0 = 1:4m*n
#   Sigma = map(x->x[idx0,:], Sigma)

#   idx = collect(1:m+1)
#   global loss = sum((U[it0:end, idx] - Uval[it0:end, idx])^2) 
# end
U = set_shape(U, (NT+1, size(U,2)))
Uval2 = matread("viscoelasticity.mat")["U"]
loss = 1e10*sum((U[end, 1:m+1]-Uval2[end, 1:m+1])^2) 

sess = Session(); init(sess)

# @show run(sess, loss)
# g = gradients(loss, pl)
# localmin = [266.8724450474673 
# 126.39699437649368]
# gradview(sess, pl, g, loss, localmin)
# savefig("line.png")

BFGS!(sess, loss)

# Uval,Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])
# matwrite("viscoelasticity.mat", Dict("U"=>Uval, "Sigma"=>Sigmaval, "Varepsilon"=>Varepsilonval))

# visualize_von_mises_stress(Sigmaval, m, n, h, name="_viscoelasticity")
# visualize_scattered_displacement(Array(Uval'), m, n, h, name="_viscoelasticity", 
#                 xlim_=[-2h, m*h+2h], ylim_=[-2h, n*h+2h])

# close("all")
# plot(LinRange(0, 20, NT+1), Uval[:,m+1], label="viscoelasticity")
# xlabel("Time")
# ylabel("Displacement")
# savefig("disp.png")

# Uval = matread("linear.mat")["U"]
# plot(LinRange(0, 20, NT+1), Uval[:,m+1], label="linear elasticity")
# legend()
# savefig("disp.png")

