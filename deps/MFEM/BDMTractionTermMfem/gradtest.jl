using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function bdm_traction_term_mfem(gd,edges)
    bdm_traction_term_mfem_ = load_op_and_grad("./build/libBdmTractionTermMfem","bdm_traction_term_mfem")
    gd,edges = convert_to_tensor(Any[gd,edges], [Float64,Int64])
    bdm_traction_term_mfem_(gd,edges)
end

# TODO: specify your input parameters
u = bdm_traction_term_mfem(gd,edges)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(bdm_traction_term_mfem(gd,edges)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10,20))
v_ = rand(10,20)
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
