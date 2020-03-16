include("utils.jl")

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
D = E/(1+ν)/(1-2ν)*[1-ν ν 0;ν 1-ν 0;0 0 1-2ν]
D = constant(D)

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


qo, qw = constant(qo), constant(qw)
function porosity(u)
    ε = compute_fvm_mechanics_term(u, m, n, h)/h^2
    ε = reshape(ε,(m, n))'
    1 - constant(1 .- φ0) .* exp(-ε)
end

################### solid equations 
function solid(Ψ2)
    p = Ψ2 #+ ρw*g*Z 
    # p = p - ave_normal(p)
    rhs = InteractionM*reshape(p, (-1,)) 
    mask = ones(2*(m+1)*(n+1))
    mask[[bdnode;bdnode .+ (m+1)*(n+1)]] .= 0.0
    rhs = rhs .* mask 
    StiffM\rhs
    # TODO:DEBUG 
    # constant(zeros(2(m+1)*(n+1)))
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
    p = upwps_op(K * K_CONST, λ, load_normal, p, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 

    # step 3: implicit transport
    sw = sat_op2(sw, dotφ, p, K * K_CONST, φ , qw[i], qo[i], μw, μo, sw, Δt, h)
    return sw, p
end
###################

function simulate()
    ta_u, ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_u = write(ta_u, 1, constant(zeros(2*(m+1)*(n+1))))
    ta_u = write(ta_u, 2, constant(zeros(2*(m+1)*(n+1))))
    ta_sw = write(ta_sw, 1, constant(sw0))
    ta_sw = write(ta_sw, 2, constant(sw0))
    ta_p = write(ta_p, 1, constant(zeros(n, m)))
    ta_p = write(ta_p, 2, constant(zeros(n, m)))
    i = constant(2, dtype=Int32)
    function condition(i, tas...)
        i <= NT
    end
    function body(i, tas...)
        ta_u, ta_sw, ta_p = tas
        u = read(ta_u, i)
        uold = read(ta_u, i-1)
        sw, p = read(ta_sw, i), read(ta_p, i)
        sw, p = fluid(i, u, uold, sw, p)
        unew = solid(p)
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        ta_u = write(ta_u, i+1, unew)
        i+1, ta_u, ta_sw, ta_p
    end
    _, ta_u, ta_sw, ta_p = while_loop(condition, body, [i, ta_u, ta_sw, ta_p])
    out_u, out_sw, out_p = stack(ta_u), stack(ta_sw), stack(ta_p)
end


u, S2, Ψ2 = simulate()
sess = Session(); init(sess)
U, s2, ψ2 = run(sess, [u, S2, Ψ2])
# visualize_saturation(s2)
# visualize_potential(ψ2)
# visualize_displacement(U*100)
# plot(U[:,5])
# visualize_stress(run(sess, D), U'|>Array, m, n, h)