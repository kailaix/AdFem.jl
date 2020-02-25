using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
np = pyimport("numpy")


mode = "training"

## alpha-scheme
β = 1/4; γ = 1/2
a = b = 0.1

n = 10
m = 2n 
h = 0.01
NT = 500
it0 = NT÷2
Δt = 1.5/NT
ηmax = 1
ηmin = 0.5

bdedge = bcedge("right", m, n, h)
bdnode = bcnode("lower", m, n, h)

# λ = Variable(1.0)
# μ = Variable(1.0)
# invη = Variable(1.0)

function eta_model(idx)
    out = ηmin * ones(4, m, n)
    out[:, :, 1:div(n,3)] .= ηmax
    out[:]
end

function visualize_inv_eta(X, k)
    x = LinRange(0.5h,m*h, m)
    y = LinRange(0.5h,n*h, n)
    V = zeros(m, n)
    for i = 1:m  
        for j = 1:n 
            elem = (j-1)*m + i 
            V[i, j] = mean(X[4(elem-1)+1:4elem])
        end
    end
    close("all")
    pcolormesh(x, y, V'/1000, vmin=ηmin-(ηmax-ηmin)/4, vmax=ηmax+(ηmax-ηmin)/4)
    colorbar(shrink=0.5)
    xlabel("x")
    ylabel("y")
    # title("Iteration = $k")
    title("True model")
    axis("scaled")
    gca().invert_yaxis()
    savefig("iter$k.png")
end

λ = constant(2.0)
μ = constant(0.5)
if mode=="data"
  global invη = constant(eta_model(1))
  global invη *= 1000.0
else
    # global invη_ = Variable(10.0*ones(4*m*n))

    # invη1d_ = Variable(10*ones(2m))
    # invη_ = repeat(invη1d_, 1, 2)[:]
    # global invη_ = repeat(invη_, n)

    invη_var = Variable((ηmin + ηmax)/2*ones(n))
    # invη1d_ = repeat(invη1d_, 1, 2)[:]
    invη_ = tf.reshape(repeat(invη_var, 1, 4m), (-1,))
    # invη_ = repeat(invη_, m)
    # global invη_ = tf.transpose(invη2d_)[:]
    global invη = 1000*invη_
end



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


M = compute_fem_mass_matrix1(m, n, h)
Zero = spzeros((m+1)*(n+1), (m+1)*(n+1))
M = SparseTensor([M Zero;Zero M])

K = compute_fem_stiffness_matrix(H, m, n, h)
C = a*M + b*K # damping matrix 
L = M + γ*Δt*C + β*Δt^2*K
L, Lbd = fem_impose_Dirichlet_boundary_condition(L, bdnode, m, n, h)


a = TensorArray(NT+1); a = write(a, 1, zeros(2(m+1)*(n+1))|>constant)
v = TensorArray(NT+1); v = write(v, 1, zeros(2(m+1)*(n+1))|>constant)
d = TensorArray(NT+1); d = write(d, 1, zeros(2(m+1)*(n+1))|>constant)
U = TensorArray(NT+1); U = write(U, 1, zeros(2(m+1)*(n+1))|>constant)
Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(4*m*n, 3)|>constant)
Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1,zeros(4*m*n, 3)|>constant)


Forces = zeros(NT, 2(m+1)*(n+1))
for i = 1:NT
  T = eval_f_on_boundary_edge((x,y)->0.1, bdedge, m, n, h)

  # if i>=NT÷2
  #   T *= 0.0
  # end
  T = [-T T]
#   T = [T T]
  rhs = compute_fem_traction_term(T, bdedge, m, n, h)

#   if i*Δt>0.5
#     rhs = zero(rhs)
#   end
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

  res = batch_matmul(invG/Δt, Sigma)
  F = compute_strain_energy_term(res, m, n, h) - K * U
  rhs = Forces[i] - Δt^2 * F

  td = d + Δt*v + Δt^2/2*(1-2β)*a 
  tv = v + (1-γ)*Δt*a 
  rhs = rhs - C*tv - K*td
  
  # rhs[[bdnode; bdnode.+(m+1)*(n+1)]] .= 0.0

  rhs = scatter_update(rhs, constant([bdnode; bdnode.+(m+1)*(n+1)]), constant(zeros(2*length(bdnode))))


  ## alpha-scheme
  a = L\rhs # bottleneck  
  d = td + β*Δt^2*a 
  v = tv + γ*Δt*a 
  U_new = d

  Varepsilon_new = eval_strain_on_gauss_pts(U_new, m, n, h)

  # res2 = squeeze(tf.matmul(tf.reshape(Varepsilon_new-Varepsilon, (size(Sigma,1), 1, 3)),
  #                       tf.matmul(invG, S)))

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

if mode!="data"
  data = matread("viscoelasticity.mat")
  global Uval,Sigmaval, Varepsilonval = data["U"], data["Sigma"], data["Varepsilon"]
  U.set_shape((NT+1, size(U, 2)))
  idx0 = 1:4m*n
  Sigma = map(x->x[idx0,:], Sigma)

  idx = 1:m+1
  global loss = sum((U[it0:end, idx] - Uval[it0:end, idx])^2) 
end

# opt = AdamOptimizer(0.1).minimize(loss)
sess = Session(); init(sess)

cb = (v, i, l)->begin
  println("[$i] loss = $l")
  inv_eta = v[1]
  visualize_inv_eta(inv_eta, i)
end

if mode=="data"
    Uval,Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])
  matwrite("viscoelasticity.mat", Dict("U"=>Uval, "Sigma"=>Sigmaval, "Varepsilon"=>Varepsilonval))

  visualize_von_mises_stress(Sigmaval, m, n, h, name="_viscoelasticity")
  visualize_scattered_displacement(Array(Uval'), m, n, h, name="_viscoelasticity", 
                  xlim_=[-2h, m*h+2h], ylim_=[-2h, n*h+2h])

  close("all")
  plot(LinRange(0, 20, NT+1), Uval[:,m+1])
  xlabel("Time")
  ylabel("Displacement")
  savefig("disp.png")

  cb([run(sess, invη)], "true", 0.0)
else


  
  @info run(sess, loss)
  v_ = []
  i_ = []
  l_ = []

  loss_ = BFGS!(sess, loss*1e10, vars=[invη], callback=cb, var_to_bounds=Dict(invη_var=>(0.1,2.0)))
end
# matwrite("R.mat", Dict("V"=>v_, "L"=>l_))

# for i = 1:1000
#   _, l, invη_ = run(sess, [opt, loss, invη])
#   @show i, l #, invη_
#   mod(i,5)==1 && cb([invη_], i, l)
# end



# ηs = 11:0.1:13
# losses = []
# for η in ηs
#    push!(losses,run(sess, loss, invη=>η))
# end
# plot(ηs, losses)


# Uval, Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])
# Uval[idx]




