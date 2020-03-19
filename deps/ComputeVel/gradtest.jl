using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function compute_vel(a,v0,psi,sigma,tau,eta)
    compute_vel_ = load_op_and_grad("./build/libComputeVel","compute_vel")
a,v0,psi,sigma,tau,eta = convert_to_tensor([a,v0,psi,sigma,tau,eta], [Float64,Float64,Float64,Float64,Float64,Float64])
    compute_vel_(a,v0,psi,sigma,tau,eta)
end


# verify 
n = 10
a = 2.0*ones(n)|>constant
x = 4.0*ones(n)|>constant
v0 = 0.001|>constant
Ψ = 2000.0*ones(n)|>constant
σ = 3.0*ones(n)|>constant
η = 2.0|>constant
s = x/2v0
inv_o = exp(-Ψ/a)
τ = a * (Ψ/a + log(s + sqrt(s*s+inv_o*inv_o)))*σ + η*x
# τ = a * asinh(*exp(Ψ/a))*σ + η*(x-u)/Δt 

# TODO: specify your input parameters
u = compute_vel(a,v0,Ψ,σ,τ,η)
sess = Session(); init(sess)
@show run(sess, u)
# error()


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    # return sum(compute_vel(a,v0,Ψ,σ,τ,η)^2)
    return sum(compute_vel(a,v0,Ψ,σ,τ,m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(n))
# v_ = rand(n)

# m_ = τ
# v_ = rand(n)

m_ = constant(rand())
v_ = rand()


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
    w_[i] = s_[i] - g_*sum(v_*dy_)
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
