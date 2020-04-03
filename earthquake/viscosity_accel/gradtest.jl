using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

function push_matrices(ii1,jj1,vv1,ii2,jj2,vv2,d)
    push_matrices_ = load_op_and_grad("./build/libPushMatrices","push_matrices")
    ii1,jj1,vv1,ii2,jj2,vv2,d = convert_to_tensor([ii1,jj1,vv1,ii2,jj2,vv2,d], [Int64,Int64,Float64,Int64,Int64,Float64,Int64])
    push_matrices_(ii1,jj1,vv1,ii2,jj2,vv2,d)
end

function visco_solve(rhs,vv, op)
    visco_solve_ = load_op_and_grad("./build/libViscoSolve","visco_solve")
    rhs,vv, op = convert_to_tensor([rhs,vv, op], [Float64,Float64, Int64])
    visco_solve_(rhs,vv, op)
end

A = sprand(10,10,0.7)
B = sprand(10,10,0.2)
A = A + B
rhs = rand(10)
op = push_matrices(find(constant(A))..., find(constant(B))..., 10)
ii, jj, vv = find(constant(B))

sol = visco_solve(rhs, vv, op)

sess = Session(); init(sess)

# @show run(sess, sol)
# @show A\rhs - run(sess, sol)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
Al = sprand(10,10,0.7)
function scalar_function(vv)
    A = constant(Al)
    B = SparseTensor(ii, jj, vv, 10, 10)
    A = A + B
    op = independent(push_matrices(find(A)..., find(B)..., 10))

    return sum(visco_solve(rhs, vv, op)^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(length(rhs)))
# v_ = rand(length(rhs))

m_ = constant(rand(length(vv)))
v_ = rand(length(vv))
y_ = scalar_function(m_)
dy_ = constant(run(sess, gradients(y_, m_)))
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
