using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function dirichlet_bd(ii,jj,vv,bd,m,n,h)
    dirichlet_bd_ = load_op_and_grad("./build/libDirichletBd","dirichlet_bd", multiple=true)
    ii,jj,vv,bd,m,n,h = convert_to_tensor([ii,jj,vv,bd,m,n,h], [Int64,Int64,Float64,Int32,Int32,Int32,Float64])
    dirichlet_bd_(ii,jj,vv,bd,m,n,h)
end



# TODO: specify your input parameters
m = 10
n = 10
ii = rand(1:2(m+1)*(n+1), 200)
jj = rand(1:2(m+1)*(n+1), 200)
vv = rand(100)
bd = rand(1:(m+1)*(n+1), 10)
h = 0.1
u = dirichlet_bd(ii,jj,vv,bd,m,n,h)
sess = Session(); init(sess)
# @show run(sess, u)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(vv)
    return sum(dirichlet_bd(ii,jj,vv,bd,m,n,h)[3]^2 + dirichlet_bd(ii,jj,vv,bd,m,n,h)[6]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(100))
v_ = rand(100)
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
