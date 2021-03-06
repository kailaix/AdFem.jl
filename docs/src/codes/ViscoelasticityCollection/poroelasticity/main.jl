using Revise
using AdFem
using PyCall
using LinearAlgebra
using ADCME
using MAT
using PyPlot
using ADCMEKit

np = pyimport("numpy")

# Domain information 
NT = 50
Δt = 1/NT
n = 15
m = 2*n 
h = 1. ./ n
bdnode = bcnode("left | right", m, n, h)
bdedge = bcedge("upper", m, n, h) # fixed pressure on the top 

b = 1.0
E = 1.0
ν = 0.35
H = E/(1+ν)/(1-2ν) * [1-ν ν 0.0;ν 1-ν 0.0;0.0 0.0 (1-2ν)/2]

Q, Prhs = compute_fvm_tpfa_matrix(ones(4*m*n), bdedge, zeros(size(bdedge,1)),m, n, h)
Q = SparseTensor(Q)
K = constant(compute_fem_stiffness_matrix(H, m, n, h))
L = SparseTensor(compute_interaction_matrix(m, n, h))
M = SparseTensor(compute_fvm_mass_matrix(m, n, h))
A = [K -b*L'
b*L/Δt 1/Δt*M-Q]
A, Abd = fem_impose_coupled_Dirichlet_boundary_condition(A, bdnode, m, n, h)
U = zeros(m*n+2(m+1)*(n+1), NT+1)
x = Float64[]; y = Float64[]
for j = 1:n+1
    for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
    end
end
    
# injection and production
injection = (div(n,2)-1)*m + 3
production = (div(n,2)-1)*m + m-3


function get_disp(SOURCE_SCALE)
    
    function condition(i, tas...)
        i<=NT
    end

    function body(i, tas...)
        ta_u, ta_ε, ta_σ = tas
        u = read(ta_u, i)
        σ0 = read(ta_σ, i)
        ε0 = read(ta_ε, i)

        g = -ε0*H
        rhs1 = compute_strain_energy_term(g, m, n, h)

        rhs1 = scatter_update(rhs1, [bdnode; bdnode .+ (m+1)*(n+1)], zeros(2length(bdnode)))
        rhs2 = zeros(m*n)
        rhs2[injection] += SOURCE_SCALE * h^2
        rhs2[production] -= SOURCE_SCALE * h^2
        rhs2 = rhs2 + b*L*u[1:2(m+1)*(n+1)]/Δt + 
                M * u[2(m+1)*(n+1)+1:end]/Δt + Prhs
        
        rhs = [rhs1;rhs2]
        o = A\rhs 

        ε = eval_strain_on_gauss_pts(o, m, n, h)
        σ = ε*H

        ta_u = write(ta_u, i+1, o)
        ta_ε = write(ta_ε, i+1, ε)
        ta_σ = write(ta_σ, i+1, σ)
        i+1, ta_u, ta_ε, ta_σ
    end

    i = constant(1, dtype=Int32)
    ta_u = TensorArray(NT+1); ta_u = write(ta_u, 1, constant(zeros(2(m+1)*(n+1)+m*n)))
    ta_ε = TensorArray(NT+1); ta_ε = write(ta_ε, 1, constant(zeros(4*m*n, 3)))
    ta_σ = TensorArray(NT+1); ta_σ = write(ta_σ, 1, constant(zeros(4*m*n, 3)))
    _, u_out, ε_out, σ_out = while_loop(condition, body, [i, ta_u, ta_ε, ta_σ])
    u_out = stack(u_out)
    u_out.set_shape((NT+1, size(u_out,2)))
    σ_out = stack(σ_out)
    ε_out = stack(ε_out)

    upper_idx = Int64[]
    for i = 1:m+1
        push!(upper_idx, (div(n,3)-1)*(m+1)+i)
        push!(upper_idx, (div(n,3)-1)*(m+1)+i + (m+1)*(n+1))
    end
    for i = 1:m 
        push!(upper_idx, (div(n,3)-1)*m+i+2(m+1)*(n+1))
    end

    u_out, σ_out
end

U, S = get_disp(500.0)

sess = Session()
init(sess)

Uval, Sval = run(sess, [U, S])

# p = visualize_displacement(Uval, m, n, h)
# saveanim(p, "u.gif")
# p = visualize_pressure(Uval[:,2(m+1)*(n+1)+1:end], m, n, h)
# saveanim(p, "p.gif")
# p = visualize_stress(H, Uval, m, n, h)
# saveanim(p, "s.gif")
# matwrite("data.mat", Dict("U"=>Uval, "S"=>Sval))

visualize_displacement(Uval[end,:], m, n, h)
savefig("poro_u.pdf")
visualize_pressure(Uval[end,2(m+1)*(n+1)+1:end], m, n, h)
savefig("poro_p.pdf")
visualize_stress(H, Uval[end,:], m, n, h)
savefig("poro_s.pdf")

