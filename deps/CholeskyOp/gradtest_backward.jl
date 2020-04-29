using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function cholesky_backward_op(A)
    cholesky_backward_op_ = load_op_and_grad("./build/libCholeskyOp","cholesky_backward_op")
    A = convert_to_tensor([A], [Float64]); A = A[1]
    L = cholesky_backward_op_(A)
end

# TODO: specify your input parameters
A = zeros(100, 6)
S = zeros(100, 9)
for i = 1:100
    l = rand(6)
    A[i,:] = l
    L =  [
        l[1] 0 0 
        l[4]    l[2] 0
          l[5] l[6]      l[3]
    ]
    S[i,:] = (L*L')'[:]
end
u = cholesky_backward_op(A)
sess = Session(); init(sess)
u_ = run(sess, u)
@show norm(u_-S)
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(cholesky_backward_op(m)^3)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(1000,6))
v_ = rand(1000,6)
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
