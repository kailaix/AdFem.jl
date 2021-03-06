using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using AdFem
Random.seed!(233)


mmesh = Mesh(10, 10, 0.1, degree=BDM1)
# TODO: specify your input parameters
alpha = constant(rand(get_ngauss(mmesh)))
β = constant(rand(get_ngauss(mmesh)))
u = compute_fem_bdm_mass_matrix(alpha,β, mmesh)
sess = Session(); init(sess)

uref = compute_fem_bdm_mass_matrix(run(sess, alpha),run(sess,β), mmesh)

run(sess, u)
@show sum(abs.(run(sess,u) - uref))
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(values(compute_fem_bdm_mass_matrix(alpha,x,mmesh))^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(get_ngauss(mmesh)))
v_ = rand(get_ngauss(mmesh))
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