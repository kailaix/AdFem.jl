using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using AdFem
Random.seed!(233)

function univariate_fem_stiffness(hmat,m,n,h)
    univariate_fem_stiffness_ = load_op_and_grad("./build/libUnivariateFemStiffness","univariate_fem_stiffness", multiple=true)
    hmat,m,n,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    univariate_fem_stiffness_(hmat,m,n,h)
end

m = 10
n = 5
h = 1.0
hmat = zeros(4*m*n, 2, 2)
for i = 1:4*m*n 
    hmat[i,:,:] = diagm(0=>ones(2))
end

# TODO: specify your input parameters
u = univariate_fem_stiffness(hmat,m,n,h)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
H = rand(2,2)
H = H+ H' 
K = compute_fem_stiffness_matrix1(H, m, n, h)
u = univariate_fem_stiffness(H,m,n,h)
A = SparseTensor(u..., (m+1)*(n+1), (m+1)*(n+1))
@show maximum(abs.(run(sess, A)-K))
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(hmat)
    return sum(univariate_fem_stiffness(hmat,m,n,h)[3]^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(4*m*n, 2, 2))
# v_ = rand(4*m*n, 2, 2)

m_ = constant(rand( 2, 2))
v_ = rand( 2, 2)
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
