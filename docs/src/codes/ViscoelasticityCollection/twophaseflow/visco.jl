include("utils.jl")

mode = "data"

# Global parameters
K_CONST =  9.869232667160130e-16 * 86400 * 1e3
ALPHA = 1.0
GRAV_CONST = 9.8    # gravity constant
SRC_CONST = 86400.0 #
n = 15
m = 30
h = 30.0 # meter
NT  = 50
Δt = 1000/NT
# Δt = 20.0 # day


ρw = 501.9
ρo = 1053.0
μw = 0.1
μo = 1.0
K_init = 20.0 .* ones(n,m) # initial guess of permeability 
g = GRAV_CONST
φ0 = 0.25 .* ones(n,m)
qw = zeros(NT+1, n, m)
qw[:,9,3] .= 0.005 * (1/h^2)/10.0 * SRC_CONST
qo = zeros(NT+1, n, m)
qo[:,9,28] .= -0.005 * (1/h^2)/10.0 * SRC_CONST
sw0 = zeros(n, m)
K = 20.0 .* ones(n,m) # millidarcy
K[8:10,:] .= 120.0 

# E = 6e9
E = 6.e9
ν = 0.35
D = E/(1+ν)/(1-2ν)*[1-ν ν 0;ν 1-ν 0;0 0 (1-2ν)/2] 


Z = zeros(n, m)
for j = 1:n 
    for i = 1:m 
        Z[j, i] = (j-0.5)*h   
    end
end

bdnode = bcnode("left | right", m, n, h)

InteractionM = constant(compute_interaction_matrix(m, n, h)')
StiffM = compute_fem_stiffness_matrix(D, m, n, h)/E
StiffM, _ = fem_impose_Dirichlet_boundary_condition_experimental(StiffM, bdnode, m, n, h)
StiffM = StiffM*E

# pl = placeholder([1.0])
# StiffM = StiffM*pl[1]

qo, qw = constant(qo), constant(qw)
function porosity(u)
    ε = compute_fvm_mechanics_term(u, m, n, h)/h^2
    ε = reshape(ε, (n, m))
    out = 1 - constant(1 .- φ0) .* exp(-ε)
end

################### solid equations 
invη = constant(1e-11*ones(4*m*n))
E = 6.e9
ν = 0.35

μ = E/(2(1+ν))
λ = E*ν/(1+ν)/(1-2ν)

fn_G = invη->begin 
  G = tensor([1/Δt+2/3*μ*invη -μ/3*invη 0.0
    -μ/3*invη 1/Δt+2/3*μ*invη 0.0
    0.0 0.0 1/Δt+μ*invη])
  invG = inv(G)
end
S = map(fn_G, invη)/Δt
Temp = tensor([2μ+λ λ 0.0
    λ 2μ+λ 0.0
    0.0 0.0 μ])
H = S*Temp
Hmat = compute_fem_stiffness_matrix(H, m, n, h)

Hmat_adj, _ = fem_impose_Dirichlet_boundary_condition_experimental(Hmat/E, bdnode, m, n, h)
Hmat_adj = Hmat_adj*E

function solid(Ψ2, σ, uold)
    
    p = Ψ2 #+ ρw*g*Z 
    # p = p - ave_normal(p)
    rhs = InteractionM*reshape(p, (-1,))

    Sσ = batch_matmul(S, σ)
    rhs = rhs + Hmat * uold - compute_strain_energy_term(Sσ, m, n, h)

    mask = ones(2*(m+1)*(n+1))
    mask[[bdnode;bdnode .+ (m+1)*(n+1)]] .= 0.0
    rhs = rhs .* mask 

    u = Hmat_adj\rhs

    εold = eval_strain_on_gauss_pts(uold, m, n, h)
    ε = eval_strain_on_gauss_pts(u, m, n, h)
    σ = Sσ + batch_matmul(H, ε - εold)
    # σ2 = batch_matmul(H, ε)
    # op = tf.print(norm(σ-σ2))
    # u = bind(u, op)

    return u, σ
end
###################

################### fluid  equations
function Krw(Sw)
    return Sw ^ 2
end

function Kro(So)
    return So ^ 2
end

function ave_normal(quantity)
    aa = sum(quantity)
    return aa/(m*n)
end

function fluid(i, u, uold, sw, p)
    # step 0: compute porosity and its rate 
    φ = porosity(u)
    φold = porosity(uold)
    dotφ = (φ-φold)/Δt

    # φ = constant(φ0) 
    # φold = constant(φ0) 
    # φ = φ * pl[1]
    # φold = φold * pl[1]
    # dotφ = tf.zeros_like(dotφ)
    # dotφ = constant(zeros(n, m))

    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    q = qw[i] + qo[i] + λw/(λo+1e-16).*qo[i]
    # q = qw + qo
    potential_c = (ρw - ρo)*g .* Z

    # Step 2: implicit potential
    Θ = upwlap_op(K * K_CONST, λo, potential_c, h, constant(0.0))

    load_normal = (Θ+q/ALPHA+dotφ) - ave_normal(Θ+q/ALPHA+dotφ)

    # p = poisson_op(λ.*K* K_CONST, load_normal, h, constant(0.0), constant(1))
    p = upwps_op(K * K_CONST, λ, load_normal, tf.zeros_like(p), h, constant(0.0), constant(0)) 
    # potential p = pw - ρw*g*h 

    # step 3: implicit transport
    # sw = sat_op2(sw, dotφ, p, K * K_CONST, φ , qw[i], qo[i], μw, μo, sw, Δt, h)

    # ε = compute_fvm_mechanics_term(u, m, n, h)/h^2
    # ε = reshape(ε,(m, n))'
    # φ1 = 1 - constant(1 .- φ0) .* exp(-ε)

    sw1 = sat_op2(sw, dotφ, p, K * K_CONST , φ, qw[i], qo[i], μw, μo, sw, Δt, h) 
    # sw2 = sat_op(sw, p, K * K_CONST , φ, qw[i], qo[i], μw, μo, sw, Δt, h) 

    # op = tf.print(i, norm(gradients(sum(sw1),sw)-gradients(sum(sw2), sw)))
    # p = bind(p, op)

    # op = tf.print(i, "*", norm(sw1-sw2))#-gradients(sum(sw1), sw)))
    # p = bind(p, op)

    

    # op = tf.print(sum(sw2))
    # p = bind(p, op)
    return sw1, p, φ
end
###################

function simulate()
    ta_u, ta_sw, ta_p, ta_φ, ta_σ = 
        TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_u = write(ta_u, 1, constant(zeros(2*(m+1)*(n+1))))
    ta_u = write(ta_u, 2, constant(zeros(2*(m+1)*(n+1))))
    ta_sw = write(ta_sw, 1, constant(sw0))
    ta_sw = write(ta_sw, 2, constant(sw0))
    ta_p = write(ta_p, 1, constant(zeros(n, m)))
    ta_p = write(ta_p, 2, constant(zeros(n, m)))
    ta_φ = write(ta_φ, 1, constant(φ0))
    ta_φ = write(ta_φ, 2, constant(φ0))
    ta_σ = write(ta_σ, 1, constant(zeros(4*m*n, 3)))
    ta_σ = write(ta_σ, 2, constant(zeros(4*m*n, 3)))
    i = constant(2, dtype=Int32)
    function condition(i, tas...)
        i <= NT
    end
    function body(i, tas...)
        ta_u, ta_sw, ta_p, ta_φ, ta_σ = tas
        u = read(ta_u, i)
        uold = read(ta_u, i-1)
        σ = read(ta_σ, i)
        sw, p = read(ta_sw, i), read(ta_p, i)
        sw, p, φ = fluid(i, u, uold, sw, p)
        unew, σnew = solid(p, σ, u)
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        ta_u = write(ta_u, i+1, unew)
        ta_φ = write(ta_φ, i+1, φ)
        ta_σ = write(ta_σ, i+1, σnew)
        i+1, ta_u, ta_sw, ta_p, ta_φ, ta_σ
    end
    _, ta_u, ta_sw, ta_p, ta_φ,  ta_σ = while_loop(condition, body, [i, ta_u, ta_sw, ta_p, ta_φ, ta_σ])
    out_u, out_sw, out_p, out_φ, out_σ = stack(ta_u), stack(ta_sw), stack(ta_p), stack(ta_φ), stack(ta_σ)
    return set_shape(out_u, NT+1, 2*(m+1)*(n+1)), set_shape(out_sw, NT+1, n, m), 
            set_shape(out_p, NT+1, n, m), set_shape(out_φ, NT+1, n, m),
            set_shape(out_σ, NT+1, 4*m*n, 3)
end


u, S2, Ψ2, φ, σ = simulate()
uobs = u[:, 1:m+1]

if mode!="data"
    uobs_ = readdlm("obs.txt")
    global loss = mean((uobs-uobs_)^2)
end 

sess = Session(); init(sess)
if mode=="data"
    writedlm("obs.txt", run(sess, uobs))

    global o1, o2, o3, o4, o5 = run(sess, [u, S2, Ψ2, φ, σ])
    error("Stop")
end

BFGS!(sess, loss)

# # step 1
# @show run(sess, loss)
# # gd = gradients(loss, pl)
# # @show run(sess, gd)
# # step 2
# lineview(sess, pl, loss, [1.0;1.0], [5.0;1.5])
# gradview(sess, pl, loss, [5.0;1.5])

# # 
# # visualize_potential(o4)
visualize_displacement(10*o1[end,:], m, n, h); savefig("two_u.pdf")
visualize_saturation(o2[1,:,:], m, n, h); savefig("two_s0.pdf")
visualize_saturation(o2[25,:,:], m, n, h); savefig("two_s25.pdf")
visualize_saturation(o2[51,:,:], m, n, h); savefig("two_s51.pdf")
# # visualize_potential(o3)

plot(o1[:,1:(m+1)])