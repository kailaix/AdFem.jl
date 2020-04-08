using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function spatial_varying_tangent_elastic(mu,m,n,h,type=1)
    spatial_varying_tangent_elastic_ = load_op_and_grad("./build/libSpatialVaryingTangentElastic","spatial_varying_tangent_elastic")
    mu,m_,n_,h,type = convert_to_tensor([mu,m,n,h,type], [Float64,Int64,Int64,Float64,Int64])
    H = spatial_varying_tangent_elastic_(mu,m_,n_,h,type)
    set_shape(H, (4*m*n, 2, 2))
end

# TODO: specify your input parameters
m = 10
n = 5
h = 0.1
mu = rand(4*m*n)
H = zeros(4*m*n, 2, 2)

for i = 1:4*m*n 
    H[i,:,:] = mu[i] * diagm(0=>ones(2))
end

type = 1
u = spatial_varying_tangent_elastic(mu,m,n,h,type)
sess = Session(); init(sess)
@show run(sess, u)-H

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m_)
    return sum(spatial_varying_tangent_elastic(m_,m,n,h,type)^2)
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
