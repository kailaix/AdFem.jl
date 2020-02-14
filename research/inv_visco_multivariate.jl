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
# mode = "train"
mode = "data"

## alpha-scheme
β = 1/4; γ = 1/2
a = b = 0.1


m = 20
n = 10
h = 0.01
NT = 500
it0 = NT÷2
Δt = 20/NT
ηmax = 1
ηmin = 0.5
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

function eta_model(idx)
  if idx == 1
    out = ηmin * ones(4, m, n)
    out[:, :, 1:div(n,3)] .= ηmax
    # out[:, 1:div(m,3), :] .= 12.0
    # out[:, :, div(n,3)*2:end] .= 8.0
    out[:]
  elseif idx == 2
    out = ones(4, m, n)
    for i = 1:m 
        for j = 1:n 
            x = i*h; y = j*h 
            out[1,i,j] = (x^2 + y^2) 
            out[2,i,j] = (x^2 + y^2) 
            out[3,i,j] = (x^2 + y^2) 
            out[4,i,j] = (x^2 + y^2) 
        end
    end
    out[:]
  end
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
    pcolormesh(x, y, V', vmin=ηmin-(ηmax-ηmin)/4, vmax=ηmax+(ηmax-ηmin)/4)
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
else
    # global invη_ = Variable(10.0*ones(4*m*n))

    # invη1d_ = Variable(10*ones(2m))
    # invη_ = repeat(invη1d_, 1, 2)[:]
    # global invη_ = repeat(invη_, n)

    invη_ = Variable((ηmin + ηmax)/2*ones(n))
    # invη1d_ = repeat(invη1d_, 1, 2)[:]
    invη_ = tf.reshape(repeat(invη_, 1, 4m), (-1,))
    # invη_ = repeat(invη_, m)
    # global invη_ = tf.transpose(invη2d_)[:]
    global invη_ = invη_

    # invη1d_ = Variable(10*ones(1,2m))
    # invη_ = repeat(invη1d_, 2, 1)[:]
    # global invη_ = repeat(invη_, n, 1)

    # invη1d_ = Variable(10*ones(1,2n))
    # invη_ = repeat(invη1d_, 2, 1)[:]
    # global invη_ = repeat(invη_, m, 1)


    # global invη = repeat(invη_, 1, 4)[:]
    # global invη = [invη_[1] * ones(2*m*n); invη_[2] * ones(2*m*n)]

    # global invη_ = Variable(8.0)*ones(4*m*n)
    global invη = invη_ 
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

# error()
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
  T = [-T zeros(length(T))]
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

  # Sigma * (invG/Δt) Sigma 800x3, invG/Δt: 800x3x3
  res = squeeze(tf.matmul(tf.reshape(Sigma, (size(Sigma,1), 1, 3)),(invG/Δt)))
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

  res2 = squeeze(batch_matmul(reshape(Varepsilon_new-Varepsilon, (-1, 1, 3)), tf.matmul(invG,S)))
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
    data = matread("U.mat")
  global Uval,Sigmaval, Varepsilonval = data["U"], data["Sigma"], data["Varepsilon"]
  U.set_shape((NT+1, size(U, 2)))
  idx0 = 1:4m*n
  Sigma = map(x->x[idx0,:], Sigma)

  idx = 1:m+1
  # idx = [idx; idx .+ (m+1)*(n+1)]
  # global loss = sum((U[:,idx] - Uval[:,idx])^2)  + sum((Sigma - Sigmaval[:, idx0, :])^2) #+ sum((Varepsilon - Varepsilonval)^2)
  # global loss = sum((U - Uval)^2)  + sum((Sigma - Sigmaval[:, idx0, :])^2) #+ sum((Varepsilon - Varepsilonval)^2)
  # global loss = sum((U[:,idx] - Uval[:,idx])^2) #+ sum((Varepsilon - Varepsilonval)^2)
  # global loss = sum((U - Uval)^2) 
  # global loss = sum((U[it0:end, :] - Uval[it0:end, :])^2) 
  global loss = sum((U[it0:end, idx] - Uval[it0:end, idx])^2) 
  # global loss = sum((Sigma - Sigmaval[:, idx0, :])^2)
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
  matwrite("U.mat", Dict("U"=>Uval, "Sigma"=>Sigmaval, "Varepsilon"=>Varepsilonval))
  # visualize_scattered_displacement(U, m, n, h; name = "_eta$η", xlim_=[-0.01,0.5], ylim_=[-0.05,0.22])
    # # visualize_displacement(U, m, n, h;  name = "_viscoelasticity")
    # # visualize_stress(H, U, m, n, h;  name = "_viscoelasticity")

    close("all")

    figure(figsize=(15,5))
    subplot(1,3,1)
    idx = div(n,2)*(m+1) + m+1
    # idx = (m+1)÷2
    # plot((it0-1:NT)*Δt, Uval[it0:end, idx])
    plot((0:NT)*Δt, Uval[:, idx])
    xlabel("time")
    ylabel("\$u_x\$")
    plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    ax = plt.gca().inset_axes([0.3, 0.2, 0.6, 0.7])
    ax.plot((it0-1:NT)*Δt, Uval[it0:end, idx])
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    subplot(1,3,2)
    idx = 4*(div(n,2)*m + m)
    # idx = (m+1)*2
    # plot((it0-1:NT)*Δt, Sigmaval[it0:end,idx,1])
    plot((0:NT)*Δt, Sigmaval[:,idx,1])
    xlabel("time")
    ylabel("\$\\sigma_{xx}\$")
    plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    ax = plt.gca().inset_axes([0.4, 0.1, 0.5, 0.5])
    ax.plot((it0-1:NT)*Δt, Sigmaval[it0:end,idx,1])
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    subplot(1,3,3)
    idx = 4*(div(n,2)*m + m)
    # idx = (m+1)*2
    # plot((it0-1:NT)*Δt, Varepsilonval[it0:end,idx,1])
    plot((0:NT)*Δt, Varepsilonval[:,idx,1])
    xlabel("time")
    ylabel("\$\\varepsilon_{xx}\$")
    plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    ax = plt.gca().inset_axes([0.3, 0.2, 0.6, 0.7])
    ax.plot((it0-1:NT)*Δt, Varepsilonval[it0:end,idx,1])
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    savefig("visco_eta.png")

    cb([run(sess, invη)], "true", 0)
  error()
end


@info run(sess, loss)
v_ = []
i_ = []
l_ = []

loss_ = BFGS!(sess, loss*1e10, vars=[invη], callback=cb, var_to_bounds=Dict(invη_=>(5.0,15.0)))
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




