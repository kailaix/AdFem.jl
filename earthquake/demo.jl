include("utils.jl")
using MAT
using PyPlot

## debug
meta = matopen("data.mat")
data = read(meta, "d")
time = data["time"]
v = data["slipVel"]
psi = data["psi"]
bd_right = data["momBal"]["bcR"]
bd_left = data["momBal"]["bcL"]
nt = 601
nx = 51
ny = 51
eq_ind = collect(1:nt)[v[1,:] .> 1e-1]

# bd_right = (bd_right .- bd_right[:, 1])/5 .+ bd_right[:, 1]
# bd_left = (bd_left .- bd_left[:, 1])/5 .+ bd_left[:, 1]

# figure()
# subplot(411)
# plot(time, v[1,:], label="slip velocity")
# legend()
# subplot(412)
# plot(time, psi[1,:], label=L"$\psi$")
# legend()
# subplot(413)
# plot(time, bd_left[1,:], label="slip left boundary")
# legend()
# subplot(414)
# plot(time, bd_right[1,:], label="slip right boundary")
# legend()
# tight_layout()
# savefig("data_all.png")

# figure()
# subplot(411)
# plot(time[eq_ind], v[1,eq_ind], label="slip velocity")
# legend()
# subplot(412)
# plot(time[eq_ind], psi[1,eq_ind], label=L"$\psi$")
# legend()
# subplot(413)
# plot(time[eq_ind], bd_left[1,eq_ind], label="slip left boundary")
# legend()
# subplot(414)
# plot(time[eq_ind], bd_right[1,eq_ind], label="slip right boundary")
# legend()
# tight_layout()
# savefig("data_Eq.png")


m = 50
n = 50
h = 0.1

right = bcnode("right", m, n, h)
left = bcnode("left", m, n, h)
ind = [right;left]
ρ = 3
μ = 30
η = 1e-13
a = ones(n+1) * 0.01
b = ones(n+1) * 0.02
# b[1:n÷2] .= 0.03
# b[n÷2:end] .= 0.1
Dc = 0.01
v0 = 1e-9
f0 = 0.6
Ψ0 = ones(n+1) * 1.01 * f0

# Δt = time[eq_ind[2]] - time[eq_ind[1]]



σn = ones(n+1) * 50 

Δt = time[eq_ind[2:end]] - time[eq_ind[1:end-1]]
Δt = constant(Δt)
NT = length(Δt)


# right = bcnode("right", m, n, h)
# left = bcnode("left", m, n, h)
# ind = [right;left]
# ρ = 1. 
# μ = 1.
# η = 1.
# a = constant(ones(n+1) )
# b = constant(2*ones(n+1) )
# # b[1:n÷2] .= 0.03
# # b[n÷2:end] .= 0.1
# Dc = 1.
# v0 = 1.e-3
# f0 = 1.
# Ψ0 = ones(n+1) * 1. * f0
# Δt = 10.
# NT = 100
# σn = 10. *ones(n+1) 


K_ = compute_fem_stiffness_matrix1(diagm(0=>ones(2)), m, n, h)
K_[ind, :] .= 0.0
K_[ind, ind] = spdiagm(0=>ones(length(ind)))
K_ = dropzeros(K_)
K = constant(K_)

u_bd = zeros(NT+1, n+1)
for i = 1:NT+1
    # u_bd[i, :] = (i-1)*Δt*v0 .+ bd_right[:, eq_ind[1]]
    u_bd[i, :] = bd_right[:, eq_ind[i]]
end
u_bd = constant(u_bd)

u3_bds = zeros(NT+1, n+1)
for i = 1:NT+1
    u3_bds[i, :] = bd_left[:, eq_ind[i]]
end
u3_bds = constant(u3_bds)

v3_true = constant(Array(v[:,eq_ind]'))

σzx_id = Int64[]
for i = 1:n 
    elem = (i-1)*m + 1
    push!(σzx_id, (elem-1)*4 + 1)
end
push!(σzx_id, σzx_id[end]+2)

# U0 = zeros((m+1)*(n+1))
# # U1 = v0 * Δt * ones((m+1)*(n+1))

# U1 = zeros(m+1, n+1)
# for i = 1:n+1
#     U1[end,i] =  v0 * Δt + bd_right[i, eq_ind[1]]
#     U1[1,i] = bd_left[i, eq_ind[1]]
# end
# U1 = U1[:]
# U1 = K_\U1

sess = Session(); 
u0 = vector([left; right], [bd_left[:,eq_ind[1]]; bd_right[:,eq_ind[1]]], (m+1)*(n+1))
U0 = K_\run(sess, u0)

u1 = vector([left; right], [bd_left[:,eq_ind[2]]; bd_right[:,eq_ind[2]]], (m+1)*(n+1))
U1 = K_\run(sess, u1)

Ψ0 = psi[:, eq_ind[1]]
Ψ1 = psi[:, eq_ind[2]]



# error()
#   
#   -----> x
#  |
#  ^ y 
function simulate()
    function one_step(u3, u3_old, Ψ, i)

        # rate and state friction
        ε = eval_strain_on_gauss_pts1(u3, m, n, h) # 4mn x 3
        σ = -μ * ε
        σzx = σ[σzx_id,1] # σx

        # op = tf.print("sigma zx", σzx)
        # i = bind(i, op)

        v3 = (u3[left] - u3_old[left])/Δt[i]
        # v3 = v3_true[i]


        f = a * tf.asinh(v3/(2v0)*exp(Ψ/a))
        Ψ_new = -v3/Dc*(f-f0+(b-a)*log(v3/v0)) * Δt[i] + Ψ
        Ψ_new = tf.cond(tf.equal(i,2), ()->constant(Ψ0), ()->Ψ_new)
        # op = tf.print(i, Ψ_new)
        # i = bind(i, op)

        u3_bd = rate_state_friction(a, u3[left], v0, Ψ_new, σn, σzx, η, Δt[i])
    
        # op = tf.print(i, u3_bd, u3[left])
        # i = bind(i, op)

        u3_bd = u3_bds[i+1]
        u0 = vector([right;left], [u_bd[i+1]; u3_bd], (m+1)*(n+1))

        # Ψ_new = Ψ
        # u0 = vector([bcnode("left", m, n, h); bcnode("right", m, n, h)], [constant(zeros(n+1)); u_bd[i+1]], (m+1)*(n+1))
        
        
        u3_new = K\u0
    
        return u3_new, Ψ_new
    end
    
    function body(i, ta_u3, ta_Ψ)
        u3 = read(ta_u3, i)
        u3_old = read(ta_u3, i-1)
        Ψ = read(ta_Ψ, i)
        u3_new, Ψ_new = one_step(u3, u3_old, Ψ, i)

        # op = tf.print(u3_new)
        # i = bind(i, op)

        ta_u3 = write(ta_u3, i+1, u3_new)
        ta_Ψ = write(ta_Ψ, i+1, Ψ_new)
        op = tf.print(i)
        i = bind(i, op)
        return i+1, ta_u3, ta_Ψ
    end
    
    function condition(i, ta_u3, ta_Ψ)
        i<=NT
    end

    ta_u3, ta_Ψ = TensorArray(NT+1), TensorArray(NT+1)
    i = constant(2, dtype=Int32)
    ta_u3 = write(ta_u3, 1, constant(U0))
    ta_u3 = write(ta_u3, 2, constant(U1))
    ta_Ψ = write(ta_Ψ, 1, constant(Ψ0))
    ta_Ψ = write(ta_Ψ, 2, constant(Ψ1))

    _, u3, Ψ = while_loop(condition,body, [i, ta_u3, ta_Ψ])
    u3, Ψ = stack(u3), stack(Ψ)
    return u3, Ψ
end


disp, state = simulate()
init(sess)
DISP, STATE = run(sess, [disp,state]);
plot(time[eq_ind], STATE[:,1])
# plot(time[eq_ind], DISP[:, 1])
# close("all")
# plot(U3[:,left[length(left)÷2]])
# nothing