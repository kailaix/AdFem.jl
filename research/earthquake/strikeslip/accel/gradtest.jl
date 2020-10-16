using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using AdFem
using SparseArrays
Random.seed!(233)

function fast_solve(rhs)
    fast_solve_ = load_op_and_grad("./build/libFastSolve","fast_solve")
    rhs = convert_to_tensor(rhs, dtype = Float64)
    fast_solve_(rhs)
end

m = 10
n = 5
h = 0.1


K_ = compute_fem_stiffness_matrix1(diagm(0=>ones(2)), m, n, h)
ind = collect(1:m+1)
K_[ind, :] .= 0.0
K_[ind, ind] = spdiagm(0=>ones(length(ind)))
rhs = Float64.(collect((1:(m+1)*(n+1))))
x = K_\rhs
ccall((:factorize_matrix, "./build/libSaver"), Cvoid, (Cint, Cint, Cdouble, Ref{Cint}, Cint), 
            Int32(m), Int32(n), h, Int32.(ind .- 1) , Int32(length(ind)))

tf_x = fast_solve(constant(rhs))
sess = Session(); init(sess)
@show norm(run(sess, tf_x)-x)
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(fast_solve(m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand((m+1)*(n+1)))
v_ = rand((m+1)*(n+1))
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
