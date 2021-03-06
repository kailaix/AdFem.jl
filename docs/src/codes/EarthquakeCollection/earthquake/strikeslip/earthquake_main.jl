include("utils.jl")
using MAT
using PyPlot
using ADCMEKit

## Load inital state from outside simulation Scycle
meta = matopen("data-scycle.mat")
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
eq_ind = collect(190:200)


## model setup
m = 50
n = 50
h = 0.1

right = bcnode("right", m, n, h)
left = bcnode("left", m, n, h)
ind = [right;left]
ρ = 3
μ = 30
VPL = 1e-9 # plat loading velocity
f0 = 0.6

# Initial data 
V0 = v[:, eq_ind[1]]
Ψ0 = psi[:, eq_ind[1]]
# Ψ0 = ones(n+1) * 1.01 * f0

η = constant(4.7434)
a = constant(ones(n+1) * 0.01)
b = constant(ones(n+1) * 0.02)
Dc = constant(0.03)
v0 = constant(1e-6)
f0 = constant(f0)
σn = constant(ones(n+1) * 50)
bd_left0 = constant(bd_left[:,eq_ind[1]])

function RangeVariable(a, b, n=nothing)
    if isnothing(n)
        (b-a)*sigmoid(Variable(0.0)) + a 
    else
        (b-a)*sigmoid(Variable(zeros(n))) + a 
    end
end

# η = RangeVariable(0.0,2.0) * 10.0
# a = RangeVariable(0.0, 2.0, n+1) * 0.02
# b = RangeVariable(0.0, 2.0, n+1) * 0.03
# Dc = RangeVariable(0.0,2.0)
# v0 = exp(-RangeVariable(0.0,8.0)*log(10)) 
# f0 = RangeVariable(0.0,2.0) * 0.8
# σn = RangeVariable(0.0, 2.0, n+1) * 30
# bd_left0 = RangeVariable(0.0, 2.0, n+1) * 0.3

# pl = placeholder(ones(1))
Δt = 0.01
t = collect(0:Δt:time[eq_ind[end]]-time[eq_ind[1]])
NT = length(t)
Δt = constant(ones(NT) .* Δt)

# @show NT
# NT = 5

# copy matrix to C++
ccall((:factorize_matrix, "./accel/build/libSaver"), Cvoid, (Cint, Cint, Cdouble, Ref{Cint}, Cint), 
            Int32(m), Int32(n), h, Int32.(ind .- 1) , Int32(length(ind)))

function fast_solve(rhs)
    fast_solve_ = load_op_and_grad("./accel/build/libFastSolve","fast_solve")
    rhs = convert_to_tensor(rhs, dtype = Float64)
    fast_solve_(rhs)
end

K_ = compute_fem_stiffness_matrix1(diagm(0=>ones(2)), m, n, h)
K_[ind, :] .= 0.0
K_[ind, ind] = spdiagm(0=>ones(length(ind)))
K_ = dropzeros(K_)

## DEBUG
# u_bd = zeros(NT+1, n+1)
# for i = 1:NT+1
#     u_bd[i, :] = (i-1)*fix_Δt*VPL .+ bd_right[:, eq_ind[1]]
#     # u_bd[i, :] = bd_right[:, eq_ind[i]]
# end
# u_bd = constant(u_bd)

σzx_id = Int64[]
for i = 1:n 
    elem = (i-1)*m + 1
    push!(σzx_id, (elem-1)*4 + 1)
end
push!(σzx_id, σzx_id[end]+2)

sess = Session(); 

# u0 = vector([left; right], [bd_left[:,eq_ind[1]]; bd_right[:,eq_ind[1]]], (m+1)*(n+1))
# U0 = K_\run(sess, u0)

# bd_left0 = Variable(bd_left[:,eq_ind[1]])
u0 = vector([left; right], [bd_left0; constant(bd_right[:,eq_ind[1]])], (m+1)*(n+1))
U0 = constant(K_)\u0

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
function rk3(i, u, psi, v_bd, Δt)
    ## --- RK Stage 1 ---
    du_bd_S1 = v_bd/2  # t 

    # v_bd = v_ref[i] ## debug
    
    dpsi_S1 = StateRate(psi, v_bd) # t
    u_left_bd_S1 = u[left] + 0.5*Δt*du_bd_S1 # t+0.5Δt
    psi_S1 = psi + 0.5*Δt * dpsi_S1 # t+0.5Δt

    # psi_S1 = ψ_ref[i] ## debug
    # u_left_bd_S1 = bd_left_ref[i] ## debug

    u_right_bd_S1 = u[right] + 0.5*Δt*VPL # t + 0.5Δt
    u0 = vector([left;right], [u_left_bd_S1; u_right_bd_S1], (m+1)*(n+1))
    # u_S1 = K\u0 # t + 0.5Δt
    u_S1 = fast_solve(u0)

    ε = eval_strain_on_gauss_pts1(u_S1, m, n, h)
    σ = μ * ε
    τ = σ[σzx_id ,1] 

    # τ = tau_ref[i] ## debug

    v_bd_S1 = compute_vel(a, v0, psi_S1, σn, τ, η)

    ## --- RK Stage 2 ---
    du_bd_S2 = v_bd_S1/2  # t + 0.5Δt

    # v_bd_S1 = v_ref[i] ## debug
    dpsi_S2 = StateRate(psi_S1, v_bd_S1) # t + 0.5Δt
    u_left_bd_S2 = u[left] + Δt * (-du_bd_S1 + 2*du_bd_S2) # t + Δt
    psi_S2 = psi + Δt * (-dpsi_S1 + 2*dpsi_S2) # t + Δt

    # psi_S2 = ψ_ref[i] ## debug
    # u_left_bd_S2 = bd_left_ref[i] ## debug

    u_right_bd_S2 = u[right] + Δt*VPL # t + Δt
    u0 = vector([left;right], [u_left_bd_S2; u_right_bd_S2], (m+1)*(n+1))# t + Δt
    # u_S2 = K\u0 # t + Δt
    u_S2 = fast_solve(u0)
    
    ε = eval_strain_on_gauss_pts1(u_S2, m, n, h)
    σ = μ * ε
    τ = σ[σzx_id,1] 

    ## debug
    # τ = tau_ref[i]
    # op = tf.print(τ)
    # i = bind(i, op)

    v_bd_S2 = compute_vel(a, v0, psi_S2, σn, τ, η) # t+ Δt + δΔt
    
    ## --- RK Stage 3 ---
    du_bd_S3 = v_bd_S2/2 # t + Δt

    # v_bd_S2 = v_ref[i] ## debug
    dpsi_S3  = StateRate(psi_S2, v_bd_S2) # t + Δt
    
    u_left_bd_S3 = u[left] + Δt/6*(du_bd_S1 + 4*du_bd_S2 + du_bd_S3); ## t + Δt
    psi_S3 = psi + Δt/6*(dpsi_S1 + 4*dpsi_S2 + dpsi_S3); ## t + Δt

    u_right_bd_S3 = u[right] + Δt*VPL # t + Δt
    u0 = vector([left;right], [u_left_bd_S3; u_right_bd_S3], (m+1)*(n+1))# t + Δt
    # u_S3 = K\u0 # t + Δt
    u_S3 = fast_solve(u0)

    ε = eval_strain_on_gauss_pts1(u_S3, m, n, h) 
    σ = μ * ε
    τ = σ[σzx_id,1]  

    # τ = tau_ref[i+1] ## debug
    # psi_S3 = ψ_ref[i+1] ## debug

    v_bd_S3 = compute_vel(a, v0, psi_S3, σn, τ, η) # t+ Δt + δΔt

    # psi_S3 = ψ_ref[i+1] ## debug
    # u_left_bd = bd_left_ref[i+1]  ## debug

    return u_S3, psi_S3, v_bd_S3
end



function simulate()
    
    function body(i, ta_u, ta_Ψ, ta_v_bd)
        u = read(ta_u, i)
        Ψ = read(ta_Ψ, i)
        v_bd = read(ta_v_bd, i)
        u, Ψ, v_bd = rk3(i, u, Ψ, v_bd, Δt[i])
        ta_u = write(ta_u, i+1, u)
        ta_Ψ = write(ta_Ψ, i+1, Ψ)
        # ta_Ψ = write(ta_Ψ, i+1, ψ_ref[i+1]) ## debug
        ta_v_bd = write(ta_v_bd, i+1, v_bd)
        return i+1, ta_u, ta_Ψ, ta_v_bd
    end
    
    function condition(i, ta_u, ta_Ψ, ta_v_bd)
        i<=NT
    end

    ta_u, ta_Ψ, ta_v_bd = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    i = constant(1, dtype=Int32)
    ta_u = write(ta_u, 1, convert_to_tensor(U0))
    # ta_u = write(ta_u, 1, constant(U0))
    ta_Ψ = write(ta_Ψ, 1, constant(Ψ0))
    ta_v_bd = write(ta_v_bd, 1, constant(V0))

    _, u, Ψ, v_bd = while_loop(condition,body, [i, ta_u, ta_Ψ, ta_v_bd])
    u, Ψ, v_bd = stack(u) , stack(Ψ),  stack(v_bd)
    u = set_shape(u, (NT+1, size(u,2)))
    return u, Ψ, v_bd
end


## debug
# using PyCall 
# py"""
# import traceback
# try:
#     # Your codes here 
#     $sess.run($u)
# except Exception:
#     print(traceback.format_exc())
# """

