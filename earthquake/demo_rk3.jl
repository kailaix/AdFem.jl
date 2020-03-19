include("utils.jl")
using MAT
using PyPlot

## debug
meta = matopen("data.mat")
data = read(meta, "d")
time = data["time"]
tau = data["tauP"]
v = data["slipVel"]
psi = data["psi"]
bd_right = data["momBal"]["bcR"]
bd_left = data["momBal"]["bcL"]
nt = 601
nx = 51
ny = 51
# eq_ind = collect(1:nt)[(v[1,:] .> 1e-1)[:]]
eq_ind = collect(220:460)
# eq_ind = collect(180:460)

m = 50
n = 50
h = 0.1

right = bcnode("right", m, n, h)
left = bcnode("left", m, n, h)
ind = [right;left]
ρ = 3
μ = 30
η = 4.7434 
# η = 4.7434
a = constant(ones(n+1) * 0.01)
b = constant(ones(n+1) * 0.02)
# b[1:n÷2] .= 0.03s
# b[n÷2:end] .= 0.1
Dc = 0.03
v0 = 1e-6
VPL = 1e-9 # plat loading velocity
f0 = 0.6
Ψ0 = ones(n+1) * 1.01 * f0

# Δt = time[eq_ind[2]] - time[eq_ind[1]]



σn = ones(n+1) * 50 

Δt = time[eq_ind[2:end]] - time[eq_ind[1:end-1]]
# NT = length(Δt) + 1000
# NT = 400
# Δt = constant(Δt)/10
fix_Δt = minimum(Δt)
t = collect(0:fix_Δt:time[eq_ind[end]]-time[eq_ind[1]])
NT = length(t)
Δt = constant(ones(NT) .* fix_Δt)
# Δt = constant(Δt)


# NT = 3

K_ = compute_fem_stiffness_matrix1(diagm(0=>ones(2)), m, n, h)
K_[ind, :] .= 0.0
K_[ind, ind] = spdiagm(0=>ones(length(ind)))
K_ = dropzeros(K_)
K = constant(K_)

u_bd = zeros(NT+1, n+1)
for i = 1:NT+1
    u_bd[i, :] = (i-1)*fix_Δt*VPL .+ bd_right[:, eq_ind[1]]
    # u_bd[i, :] = bd_right[:, eq_ind[i]]
end
u_bd = constant(u_bd)

# u3_bds = zeros(NT+1, n+1)
# for i = 1:NT+1
#     u3_bds[i, :] = bd_left[:, eq_ind[i]]
# end
# u3_bds = constant(u3_bds)

# v3_true = constant(Array(v[:,eq_ind]'))

σzx_id = Int64[]
for i = 1:n 
    elem = (i-1)*m + 1
    push!(σzx_id, (elem-1)*4 + 1)
end
push!(σzx_id, σzx_id[end]+2)


# initial data 

V0 = v[:, eq_ind[1]]
V1 = v[:, eq_ind[1]]

sess = Session(); 
u0 = vector([left; right], [bd_left[:,eq_ind[1]]; bd_right[:,eq_ind[1]]], (m+1)*(n+1))
U0 = K_\run(sess, u0)

u1 = vector([left; right], [bd_left[:,eq_ind[2]]; bd_right[:,eq_ind[2]]], (m+1)*(n+1))
U1 = K_\run(sess, u0)

Ψ0 = psi[:, eq_ind[1]]
Ψ1 = psi[:, eq_ind[1]]

ψ_ref = constant(Array(psi[:,eq_ind]'))
bd_left_ref = constant(Array(bd_left[:,eq_ind]'))
tau_ref = constant(Array(tau[:,eq_ind]'))
v_ref = constant(Array(v[:,eq_ind]'))

# du is defined on the left side
function StateRate(Ψ, v)
    # v = abs(v)
    # s = v/(2v0)
    # o = exp(-Ψ/a)
    # f = a * (Ψ/a + log(s+sqrt(s^2+o^2)))
    f = a* tf.asinh(v/(2v0)*exp(Ψ/a))
    Δψ = -v/Dc*(f-f0+(b-a)*log(v/v0))
end

# https://acadpubl.eu/jsi/2015-101-5-6-7-8/2015-101-8/21/21.pdf
function rk3(i, psi, v_bd, u, Δt)
    ## --- RK Stage 1 ---
    du_bd_S1 = v_bd/2  # t 

    # v_bd = v_ref[i]
    dpsi_S1 = StateRate(psi, v_bd) # t
    u_left_bd_S1 = u[left] + 0.5*Δt*du_bd_S1 # t+0.5Δt
    psi_S1 = psi + 0.5*Δt * dpsi_S1 # t+0.5Δt

    # psi_S1 = ψ_ref[i]
    # u_left_bd_S1 = bd_left_ref[i]

    u_right_bd_S1 = u[right] + 0.5*Δt*VPL # t + 0.5Δt
    u0 = vector([left;right], [u_left_bd_S1; u_right_bd_S1], (m+1)*(n+1))
    u_S1 = K\u0 # t + 0.5Δt

    ε = eval_strain_on_gauss_pts1(u_S1, m, n, h)
    σ = μ * ε
    τ = σ[σzx_id ,1] 

    # τ = tau_ref[i]

    v_bd_S1 = rate_state_friction(a, v0, psi_S1, σn, τ, η) # t + 0.5Δt + δΔt, 0.5 can be any positive number

    ## --- RK Stage 2 ---
    du_bd_S2 = v_bd_S1/2  # t + 0.5Δt

    # v_bd_S1 = v_ref[i]
    dpsi_S2 = StateRate(psi_S1, v_bd_S1) # t + 0.5Δt
    u_left_bd_S2 = u[left] + Δt * (-du_bd_S1 + 2*du_bd_S2) # t + Δt
    psi_S2 = psi + Δt * (-dpsi_S1 + 2*dpsi_S2) # t + Δt

    # psi_S2 = ψ_ref[i]
    # u_left_bd_S2 = bd_left_ref[i]

    u_right_bd_S2 = u[right] + Δt*VPL # t + Δt
    u0 = vector([left;right], [u_left_bd_S2; u_right_bd_S2], (m+1)*(n+1))# t + Δt
    u_S2 = K\u0 # t + Δt
    
    ε = eval_strain_on_gauss_pts1(u_S2, m, n, h)
    σ = μ * ε
    τ = σ[σzx_id,1] 

    # op = tf.print(τ)
    # i = bind(i, op)

    # τ = tau_ref[i]

    # op = tf.print(τ)
    # i = bind(i, op)

    

    v_bd_S2 = rate_state_friction(a, v0, psi_S2, σn, τ, η) # t+ Δt + δΔt
    
    ## --- RK Stage 3 ---
    du_bd_S3 = v_bd_S2/2 # t + Δt

    # v_bd_S2 = v_ref[i]
    dpsi_S3  = StateRate(psi_S2, v_bd_S2) # t + Δt
    
    u_left_bd_S3 = u[left] + Δt/6*(du_bd_S1 + 4*du_bd_S2 + du_bd_S3); ## t + Δt
    psi_S3 = psi + Δt/6*(dpsi_S1 + 4*dpsi_S2 + dpsi_S3); ## t + Δt

    u_right_bd_S3 = u[right] + Δt*VPL # t + Δt
    u0 = vector([left;right], [u_left_bd_S3; u_right_bd_S3], (m+1)*(n+1))# t + Δt
    u_S3 = K\u0 # t + Δt

    ε = eval_strain_on_gauss_pts1(u_S3, m, n, h)
    σ = μ * ε
    τ = σ[σzx_id,1] 

    # τ = tau_ref[i+1]
    # psi_S3 = ψ_ref[i+1]

    v_bd_S3 = rate_state_friction(a, v0, psi_S3, σn, τ, η) # t+ Δt + δΔt
    

    


    # psi_S3 = ψ_ref[i+1]
    # u_left_bd = bd_left_ref[i+1]

    # return psi_S3, v_bd_S3, u_S3

    return psi_S3, v_bd_S3, u_S3
end



function simulate()
    
    function body(i, ta_u, ta_Ψ, ta_v_bd)
        u = read(ta_u, i)
        Ψ = read(ta_Ψ, i)
        v_bd = read(ta_v_bd, i)
        Ψ, v_bd, u = rk3(i, Ψ, v_bd, u, Δt[i-1])
        ta_u = write(ta_u, i+1, u)
        ta_Ψ = write(ta_Ψ, i+1, Ψ)

        # ta_Ψ = write(ta_Ψ, i+1, ψ_ref[i+1])
        
        ta_v_bd = write(ta_v_bd, i+1, v_bd)
        return i+1, ta_u, ta_Ψ, ta_v_bd
    end
    
    function condition(i, ta_u, ta_Ψ, ta_v_bd)
        i<=NT
    end

    ta_u, ta_Ψ, ta_v_bd = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    i = constant(2, dtype=Int32)
    ta_u = write(ta_u, 1, constant(U0))
    ta_u = write(ta_u, 2, constant(U1))
    ta_Ψ = write(ta_Ψ, 1, constant(Ψ0))
    ta_Ψ = write(ta_Ψ, 2, constant(Ψ1))
    ta_v_bd = write(ta_v_bd, 1, constant(V0))
    ta_v_bd = write(ta_v_bd, 2, constant(V1))

    _, u, Ψ, v_bd = while_loop(condition,body, [i, ta_u, ta_Ψ, ta_v_bd])
    u, Ψ, v_bd = stack(u), stack(Ψ),  stack(v_bd)
    return u, Ψ, v_bd
end


disp, state, v_bd = simulate()
init(sess)
DISP, STATE, VEL = run(sess, [disp,state, v_bd]);
# DISP

# figure()
# plot(time[eq_ind], v[1, eq_ind])
# plot(time[eq_ind], VEL[:,1])
# figure()
# plot(time[eq_ind], psi[1, eq_ind])
# plot(time[eq_ind], STATE[:,1], "o--")
# figure()
# plot(time[eq_ind], bd_left[1, eq_ind])
# plot(time[eq_ind], DISP[:,1], "--")

close("all")
figure()
subplot(311)
plot(t, VEL[1:end-1,1])
subplot(312)
plot(t, STATE[1:end-1,1], "--")
subplot(313)
plot(t, DISP[1:end-1,1], "--")
suptitle("Ours")

figure()
subplot(311)
plot(time[eq_ind].-time[eq_ind[1]], v[1, eq_ind])
subplot(312)
plot(time[eq_ind].-time[eq_ind[1]], psi[1, eq_ind])
subplot(313)
plot(time[eq_ind].-time[eq_ind[1]], bd_left[1, eq_ind])
suptitle("Referece")