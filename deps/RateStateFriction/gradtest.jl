using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function rate_state_friction__(a,uold,v0,psi,sigmazx,sigmazy,eta,deltat)
    rate_state_friction_ = load_op_and_grad("./build/libRateStateFriction","rate_state_friction")
    a,uold,v0,psi,sigmazx,sigmazy,eta,deltat = convert_to_tensor([a,uold,v0,psi,sigmazx,sigmazy,eta,deltat], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64])
    rate_state_friction_(a,uold,v0,psi,sigmazx,sigmazy,eta,deltat)
end


# verify 
a = 2.0
x = 10.0
u = 2.0
Δt = 2.0
v0 = 1.0
Ψ = 1000.0
σ = 3.0
η = 2.0
τ = a * asinh((x-u)/Δt/2v0*exp(Ψ/a))*σ + η*(x-u)/Δt 

# u = 4.0
# TODO: specify your input parameters
x_est = rate_state_friction__([a],[u],v0,[Ψ],[σ], [τ], η, Δt)
sess = Session(); init(sess)
@show run(sess, x_est)

# uncomment it for testing gradients
error() 

n = 100

a = rand(n)
uold = rand(n)
v0 = 1.0
psi = rand(n)
sigmazx = rand(n)
sigmazy = rand(n)
eta = 1.0
deltat = 1.0



# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    # return sum(rate_state_friction(a,uold,v0,psi,sigmazx,sigmazy,eta,deltat)^2)
    return sum(rate_state_friction__(a,uold,v0,psi,sigmazx,sigmazy,eta,deltat)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(n))
v_ = rand(n)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 0.001 / 10^(1:5)

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
