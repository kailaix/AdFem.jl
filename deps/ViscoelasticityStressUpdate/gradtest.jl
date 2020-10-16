using Revise
using ADCME
using PyCall
using LinearAlgebra
using AdFem
using PyPlot
using Random
Random.seed!(233)


# TODO: specify your input parameters
σ1 = rand(10, 3)
σ2 = zeros(10, 3)
ε1 = rand(10, 3)
ε2 = rand(10, 3)
μ = rand(10)
η = rand(10)
λ = rand(10)
Δt = rand()

u1 = update_stress_viscosity(ε2, ε1, σ1, η, μ, λ, Δt)

u2 = update_stress_viscosity(constant(ε2), ε1, σ1, η, μ, λ, Δt)

sess = Session(); init(sess)
@show run(sess, u2)-u1

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(update_stress_viscosity(ε2, ε1, σ1, η, μ, x, Δt)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10))
v_ = rand(10)
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