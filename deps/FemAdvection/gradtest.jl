using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function fem_advection(u,v,m,n,h)
    fem_advection_ = load_op_and_grad("./build/libFemAdvection","fem_advection", multiple=true)
    u,v,m_,n_,h = convert_to_tensor(Any[u,v,m,n,h], [Float64,Float64,Int64,Int64,Float64])
    ii, jj, vv = fem_advection_(u,v,m_,n_,h)
    SparseTensor(ii+1, jj+1, vv, (m+1)*(n+1), (m+1)*(n+1))
end

m = 10
n = 10
h = 0.1
u = rand(4*m*n)
v = rand(4*m*n)

# TODO: specify your input parameters
U = fem_advection(u,v,m,n,h)
sess = Session(); init(sess)
# @show run(sess, U)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(values(fem_advection(u,x,m,n,h))^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(4*m*n))
v_ = rand(4*m*n)
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
