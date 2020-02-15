using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using ADCME
using MAT
using JLD2
using PyPlot
np = pyimport("numpy")


mode = "training2"
# Domain information 
NT = 20
Δt = 1/NT
n = 10
m = 2*n 
h = 1.0/n 
bdnode = Int64[]
for i = 1:m+1
    for j = 1:n+1
        if j==n+1
            push!(bdnode, (j-1)*(m+1)+i)
        end
    end
end


b = 1.0
invη = 1.0
λ = constant(2.0)
μ = constant(0.5)
invη = constant(invη)


if mode=="training2"
    global λ = Variable(1.5)
    global μ = Variable(1.5)
    global invη = Variable(1.5)
end
iS = tensor(
        [1+2/3*μ*Δt*invη -1/3*μ*Δt*invη 0.0
        -1/3*μ*Δt*invη 1+2/3*μ*Δt*invη 0.0 
        0.0 0.0 1+μ*Δt*invη]
    )
S = inv(iS)
H = S * tensor([
    2μ+λ λ 0.0
    λ 2μ+λ 0.0
    0.0 0.0 μ
])

Q = SparseTensor(compute_fvm_tpfa_matrix(m, n, h))
K = compute_fem_stiffness_matrix(H, m, n, h)
L = SparseTensor(compute_interaction_matrix(m, n, h))
M = SparseTensor(compute_fvm_mass_matrix(m, n, h))
A = [K -b*L'
b*L/Δt 1/Δt*M-Q]
A, Abd = fem_impose_coupled_Dirichlet_boundary_condition(A, bdnode, m, n, h)
# error()
U = zeros(m*n+2(m+1)*(n+1), NT+1)
x = Float64[]; y = Float64[]
for j = 1:n+1
    for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
    end
end
    
injection = (div(n,2)-1)*m + 3
production = (div(n,2)-1)*m + m-3


function get_disp(ipval)
    

    function condition(i, tas...)
        i<=NT
    end

    function body(i, tas...)
        ta_u, ta_ε, ta_σ = tas
        u = read(ta_u, i)
        σ0 = read(ta_σ, i)
        ε0 = read(ta_ε, i)
        rhs1 = compute_fem_viscoelasticity_strain_energy_term(ε0, σ0, S, H, m, n, h)
        rhs1 = scatter_update(rhs1, [bdnode; bdnode .+ (m+1)*(n+1)], zeros(2length(bdnode)))
        rhs2 = zeros(m*n)
        rhs2[injection] += 1.0 
        rhs2[production] -= 1.0
        rhs2 = rhs2 * ipval + b*L*u[1:2(m+1)*(n+1)]/Δt + 
                M * u[2(m+1)*(n+1)+1:end]/Δt
        
        rhs = [rhs1;rhs2]
        o = A\rhs 

        ε = eval_strain_on_gauss_pts(o, m, n, h)
        σ = σ0*S + (ε - ε0)*H
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

    idx = 1:m+1
    ux_disp = u_out[:, idx]
    ux_disp, u_out, σ_out
end

disps = []
for i = 1:5
    push!(disps, get_disp(0.2*i)[1])
end
_, test_disp, test_sigma = get_disp(0.5)

if occursin("training", mode)
    @load "invdata.jld2" Udata_
    global loss = sum([sum((disps[i] - Udata_[i])^2) for i = 1:5])
    # global opt = AdamOptimizer().minimize(loss)
end
sess = Session(); init(sess)

if mode=="data"
    Udata_ = run(sess, disps)
    @save "invdata.jld2" Udata_
    @show run(sess, [H,S])
    TDISP, Sigma = run(sess, [test_disp, test_sigma])
    visualize_scattered_displacement(TDISP'|>Array, m, n, h, name="_inv_visco_ref", 
                xlim_=[-2h, m*h+2h], ylim_=[-2h, n*h+2h])
    visualize_von_mises_stress(Sigma, m, n, h, name="_inv_visco_ref")
else
    global loss_ = BFGS!(sess, loss, 1000)
    TDISP, Sigma = run(sess, [test_disp, test_sigma])
    visualize_scattered_displacement(TDISP'|>Array, m, n, h, name="_inv_visco_test", 
                xlim_=[-2h, m*h+2h], ylim_=[-2h, n*h+2h])
    visualize_von_mises_stress(Sigma, m, n, h, name="_inv_visco_test")
end

# semilogy(loss_)
# xlabel("Iteration")
# ylabel("Loss")

# plot()
# julia> run(sess, H)
# 3×3 Array{Float64,2}:
#   2.24497    1.68432   -0.0577557
#  -0.82159    0.512678   0.484218 
#  -0.918737  -0.688694   0.361021 

# julia> run(sess, S)
# 3×3 Array{Float64,2}:
#   0.419232  -0.116582   -0.103483 
#  -0.116582   0.356347   -0.0379917
#  -0.103483  -0.0379917   0.599339 