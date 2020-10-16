using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using AdFem
using Test
Random.seed!(233)

function tpfa_op(kvalue,bc,pval,m,n,h)
    tpfa_op_ = load_op_and_grad("./build/libTpfaOp","tpfa_op", multiple=true)
    kvalue,bc,pval,m_,n_,h = convert_to_tensor(Any[kvalue,bc,pval,m,n,h], [Float64,Int64,Float64,Int64,Int64,Float64])
    ii, jj, vv, rhs = tpfa_op_(kvalue,bc,pval,m_,n_,h)
    SparseTensor(ii + 1, jj + 1, vv, m*n, m*n), set_shape(rhs, m*n)
end

# TODO: specify your input parameters
m = 20
n = 10
h = 0.1
kvalue = rand(m*n)
bc = bcedge("all", m, n, h)
pval = rand(size(bc, 1))
K, rhs = compute_fvm_tpfa_matrix(kvalue,bc,pval,m,n,h)
K1, rhs1 = tpfa_op(kvalue,bc,pval,m,n,h)
sess = Session(); init(sess)
K2 = run(sess, K1)
rhs2 = run(sess, rhs1)

@test rhs≈rhs2
@test K ≈ K2

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    # A, rhs = tpfa_op(kvalue,bc,x,m,n,h)
    # return sum(values(A)^2) + sum(rhs^2)
    A, rhs = tpfa_op(x,bc,pval,m,n,h)
    return sum(values(A)^2) + sum(rhs^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(size(bc,1)))
# v_ = rand(size(bc,1))

m_ = constant(rand(m*n))
v_ = rand(m*n)

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
