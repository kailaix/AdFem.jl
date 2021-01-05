using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using AdFem
Random.seed!(233)


# TODO: specify your input parameters
mmesh = Mesh(10, 10, 0.1)
left = -1
right = -1
cmat = rand(get_ngauss(mmesh), 2, 2)
nv = rand(get_ngauss(mmesh), 2)
u = compute_parallel_parallel_gradient_tensor(cmat, nv, mmesh)
sess = Session(); init(sess)
U = run(sess, u)

for i = 1:get_ngauss(mmesh)
    C = cmat[i,:,:]
    n = nv[i,:]
    if left == -1 
        L = I - n * n' 
    else
        L = n * n' 
    end
    if right == -1
        R = I - n * n' 
    else
        R = n * n' 
    end
    N = L * C * R 
    @info norm(N - U[i,:,:])
end
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    x = reshape(x, (get_ngauss(mmesh), 2, 2))
    out = compute_perpendicular_parallel_gradient_tensor(x, nv, mmesh)
    # out = compute_parallel_parallel_gradient_tensor(x, nv, mmesh)
    # out = compute_perpendicular_perpendicular_gradient_tensor(x, nv, mmesh)
    # out = compute_parallel_perpendicular_gradient_tensor(x, nv, mmesh)
    sum(out^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(get_ngauss(mmesh)* 2* 2))
v_ = rand(get_ngauss(mmesh)* 2* 2)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
savefig("test.png")