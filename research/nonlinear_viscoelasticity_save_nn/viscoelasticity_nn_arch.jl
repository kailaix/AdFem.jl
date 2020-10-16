using Revise
using AdFem
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using DelimitedFiles
using MAT
using Statistics
using ADCMEKit
np = pyimport("numpy")

model_id = 2
width = 1
depth = 1
activation = "tanh"
if length(ARGS)==4
  global model_id = parse(Int64, ARGS[1])
  global width = parse(Int64, ARGS[2])
  global depth = parse(Int64, ARGS[3])
  global activation = ARGS[4]
end

@show model_id, width, depth, activation

n = 15
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

# function eta_fun(σ)
#   return constant(10*ones(4*m*n)) + 5.0/(1+1000*sum(σ[:,1:2]^2, dims=2))
# end

config = width * ones(Int64, depth)
θ = Variable(ae_init([3,config...,1]))
β = 1/4; γ = 1/2
function eta_fun(σ)
  # return constant(10*ones(4*m*n)) + 5.0/(1+1000.0*sum(σ^2, dims=2))
  return ae(σ, [config...,1], θ, activation=activation)
end



# invη is a 4*m*n array 
function make_matrix(invη)
  a = b = 0.1
  fn_G = invη->begin 
    G = tensor([1/Δt+2/3*μ*invη -μ/3*invη 0.0
      -μ/3*invη 1/Δt+2/3*μ*invη 0.0
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

function simulate(FORCE_SCALE)


  a = TensorArray(NT+1); a = write(a, 1, zeros(2(m+1)*(n+1))|>constant)
  v = TensorArray(NT+1); v = write(v, 1, zeros(2(m+1)*(n+1))|>constant)
  d = TensorArray(NT+1); d = write(d, 1, zeros(2(m+1)*(n+1))|>constant)
  U = TensorArray(NT+1); U = write(U, 1, zeros(2(m+1)*(n+1))|>constant)
  Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(4*m*n, 3)|>constant)
  Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1,zeros(4*m*n, 3)|>constant)


  Forces = zeros(NT, 2(m+1)*(n+1))
  for i = 1:NT
    T = eval_f_on_boundary_edge((x,y)->0.1*FORCE_SCALE, bdedge, m, n, h)

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
    rhs = Forces[i] - F

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

  U = set_shape(U, (NT+1, size(U,2)))
  Sigma = set_shape(Sigma, (NT+1, 4*m*n, 3))

  return U, Sigma
end

function visualize(i, U0, U_, Sigma0, Sigma_)
  close("all")
  figure(figsize=(13,4))
  subplot(121)
  plot(LinRange(0, 2.0, NT+1), U0[:,1], "r--")  # reference
  plot(LinRange(0, 2.0, NT+1), U_[:,1], "r")
  plot(LinRange(0, 2.0, NT+1), U0[:,1+(n+1)*(m+1)], "g--")
  plot(LinRange(0, 2.0, NT+1), U_[:,1+(n+1)*(m+1)], "g")
  xlabel("Time")
  ylabel("Displacement")
  subplot(122)
  plot(LinRange(0, 2.0, NT+1), mean(Sigma0[:,1:4,1], dims=2)[:],"r--", label="\$\\sigma_{xx}\$")
  plot(LinRange(0, 2.0, NT+1), mean(Sigma0[:,1:4,2], dims=2)[:],"b--", label="\$\\sigma_{yy}\$")
  plot(LinRange(0, 2.0, NT+1), mean(Sigma0[:,1:4,3], dims=2)[:],"g--", label="\$\\sigma_{xy}\$")
  plot(LinRange(0, 2.0, NT+1), mean(Sigma_[:,1:4,1], dims=2)[:],"r-")
  plot(LinRange(0, 2.0, NT+1), mean(Sigma_[:,1:4,2], dims=2)[:],"b-")
  plot(LinRange(0, 2.0, NT+1), mean(Sigma_[:,1:4,3], dims=2)[:],"g-")
  legend()
  legend()
  xlabel("Time")
  ylabel("Stress")
  savefig("data/model$(model_id)_$(width)_$(depth)_$(activation)_nn$i.png")
  matwrite("data/model$(model_id)_$(width)_$(depth)_$(activation)_nn$i.mat", Dict("U"=>U_, "S"=>Sigma_))
end


U = Array{PyObject}(undef, 4)
Sigma = Array{PyObject}(undef, 4)
for (k,FORCE_SCALE) in enumerate([0.5, 0.8, 1.5, 1.0])
  U[k], Sigma[k] = simulate(FORCE_SCALE)
end
D = matread("data$model_id.mat")["U"]
S = matread("data$model_id.mat")["S"]
loss = 1e5*sum((U[4][:, 1:m+1] - D[4][:,1:m+1])^2)
sess = Session(); init(sess)
init_loss = run(sess, loss)
visualize(0, D[4], run(sess, U[4]), S[4], run(sess,Sigma[4]))
loss_ = BFGS!(sess, loss, 200)
loss_ = [init_loss;loss_]
visualize(200, D[4], run(sess, U[4]), S[4], run(sess,Sigma[4]))
writedlm("data/loss_$(model_id)_$(width)_$(depth)_$(activation).txt", reshape(loss_,:,1))
