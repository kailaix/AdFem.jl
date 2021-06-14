using Revise
using AdFem
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)


# TODO: specify your input parameters
mmesh = Mesh(10,10,0.1)


edgeid = bcedge((x1, y1, x2, y2)->(y1>0.99) && (y2>0.99), mmesh)
t0 = eval_f_on_boundary_edge((x,y)->x+y, edgeid, mmesh)
u0 = compute_fem_traction_term1(t0, edgeid, mmesh)
t = constant(t0)
u = compute_fem_traction_term1(t, edgeid, mmesh)
sess = Session(); init(sess)
@show maximum(abs.(run(sess, u) - u0))

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(compute_fem_traction_term1(m, edgeid, mmesh)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(length(t)))
v_ = rand(length(t))
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
