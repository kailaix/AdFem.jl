using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using AdFem
Random.seed!(233)

function fem_grad_mfem(u)
    fem_grad_mfem_ = load_op_and_grad(AdFem.libmfem,"fem_grad_mfem")
    u = convert_to_tensor(Any[u], [Float64]); u = u[1]
    fem_grad_mfem_(u)
end

# TODO: specify your input parameters
mesh = Mesh(2, 2, 0.5, degree=2)
ipt = ones(mesh.ndof)
u = fem_grad_mfem(ipt)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(u)
    return sum(fem_grad_mfem(u)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(mesh.ndof))
v_ = rand(mesh.ndof)
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
savefig("gradtest.png")
