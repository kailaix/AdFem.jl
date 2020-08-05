using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using PoreFlow
Random.seed!(233)

function fem_laplace_term(u,kappa,m,n,h)
    fem_laplace_term_ = load_op_and_grad("./build/libFemLaplaceTerm","fem_laplace_term")
    u,kappa,m,n,h = convert_to_tensor(Any[u,kappa,m,n,h], [Float64,Float64,Int64,Int64,Float64])
    fem_laplace_term_(u,kappa,m,n,h)
end

m = 10
n = 20
h = 0.1
u = rand((m+1)*(n+1))
kappa = rand(4*m*n)

# TODO: specify your input parameters
out = fem_laplace_term(u,kappa,m,n,h)
sess = Session(); init(sess)
res = run(sess, out)

M = compute_fem_laplace_matrix1(kappa, m, n, h)
out_ = M * u 

@info norm(res - out_)
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    # return sum(fem_laplace_term(u,kappa,m,n,h)^2)
    return sum(fem_laplace_term(u,x,m,n,h)^2)

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
