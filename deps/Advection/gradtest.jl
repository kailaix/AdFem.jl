using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

Sys.isapple() && matplotlib.use("macosx")

function advection_jl(m, n, h, v, u)
    # v: velocity, (m*n) * 2
    # u: temperature, (m*n)
    a = zeros(m*n)
    for j=1:n
        for i=1:m
            vx = v[(j-1)*m+i, 1]
            vy = v[(j-1)*m+i, 2]
            if vx >= 0
                if i==1
                    a[(j-1)*m+i] += vx * 2 * h * u[(j-1)*m+i]
                else
                    a[(j-1)*m+i] += vx * h * (u[(j-1)*m+i] - u[(j-1)*m+i-1])
                end
            else
                if i==m
                    a[(j-1)*m+i] += - vx * 2 * h * u[(j-1)*m+i]
                else
                    a[(j-1)*m+i] += vx * h * (u[(j-1)*m+i+1] - u[(j-1)*m+i])
                end
            end
                

            if vy >= 0
                if j==1
                    a[(j-1)*m+i] += vy * 2 * h * u[(j-1)*m+i]
                else
                    a[(j-1)*m+i] += vy * h * (u[(j-1)*m+i] - u[(j-2)*m+i])
                end
            else
                if j==n
                    a[(j-1)*m+i] += - vy * 2 * h * u[(j-1)*m+i]
                    
                else
                    a[(j-1)*m+i] += vy * h * (u[j*m+i] - u[(j-1)*m+i])
                end
            end
        end
    end
    return a
end

function advection(v,u,m,n,h)
    advection_ = load_op_and_grad("./build/libAdvection","advection")
    v,u,m,n,h = convert_to_tensor(Any[v,u,m,n,h], [Float64,Float64,Int64,Int64,Float64])
    advection_(v,u,m,n,h)
end

# TODO: specify your input parameters
m = 10
n = 10
h = 0.1
v = rand(m*n,2)
u = rand(m*n)
jl_u = advection_jl(m, n, h, v, u)
u = advection(v,u,m,n,h)
sess = Session(); init(sess)
@show run(sess, u)-jl_u

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(advection(x,u,m,n,h)^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(m*n))
# v_ = rand(m*n)

m_ = constant(rand(m*n,2))
v_ = rand(m*n,2)
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
savefig("gradv.png")