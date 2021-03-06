using Revise
using AdFem
using PyCall
using LinearAlgebra
using ADCME
using MAT
using JLD2
using DelimitedFiles
using PyPlot
np = pyimport("numpy")
mpl = pyimport("tikzplotlib")


mode = "training"
# Domain information 
NT = 20
Δt = 1/NT
n = 10
m = 2*n 
h = 1.0/n 
bdnode = bcnode("lower", m, n, h)


b = 1.0
# pl = placeholder([2.0;0.5;1.0])
pl = Variable(0.8*ones(3))
λ, μ, invη = pl[1], pl[2], pl[3]

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

bd = bcedge("upper", m, n, h)
Q, Prhs = compute_fvm_tpfa_matrix(ones(4*m*n), bd, zeros(size(bd,1)),m, n, h)
Q = SparseTensor(Q)
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
                M * u[2(m+1)*(n+1)+1:end]/Δt + Prhs
        
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

    u_out, σ_out
end

Us = Array{PyObject}(undef, 6)
Ss = Array{PyObject}(undef, 6)

for i = 1:5
    Us[i], Ss[i] = get_disp(0.2*i)
end
Us[6], Ss[6] = get_disp(0.5)

if isfile("invdata.mat")
    d = matread("invdata.mat")
    Us_ = d["U"]
    Ss_ = d["S"]
    global Sigma0 = Ss_[6]
    global U0 = Us_[6]
    global loss = sum([sum((Us[i][:,1:m+1] - Us_[i][:,1:m+1])^2) for i = 1:5])
end
sess = Session(); init(sess)

function visualize(i)
    visualize_von_mises_stress(Sigma_, m, n, h, name="_nn$i")
    visualize_displacement(Array(U_'), m, n, h, name="_nn0$i", 
                    xlim_=[-3h, m*h+2h], ylim_=[-2h, n*h+2h])
    close("all")
    figure(figsize=(13,4))
    subplot(121)
    plot(LinRange(0, 20, NT+1), U0[:,1], "r--")
    plot(LinRange(0, 20, NT+1), U_[:,1], "r")
    plot(LinRange(0, 20, NT+1), U0[:,1+(n+1)*(m+1)], "g--")
    plot(LinRange(0, 20, NT+1), U_[:,1+(n+1)*(m+1)], "g")
    xlabel("Time")
    ylabel("Displacement")
    subplot(122)
    plot(LinRange(0, 20, NT+1), mean(Sigma0[:,1:4,1], dims=2)[:],"r--", label="\$\\sigma_{xx}\$")
    plot(LinRange(0, 20, NT+1), mean(Sigma0[:,1:4,2], dims=2)[:],"b--", label="\$\\sigma_{yy}\$")
    plot(LinRange(0, 20, NT+1), mean(Sigma0[:,1:4,3], dims=2)[:],"g--", label="\$\\sigma_{xy}\$")
    plot(LinRange(0, 20, NT+1), mean(Sigma_[:,1:4,1], dims=2)[:],"r-")
    plot(LinRange(0, 20, NT+1), mean(Sigma_[:,1:4,2], dims=2)[:],"b-")
    plot(LinRange(0, 20, NT+1), mean(Sigma_[:,1:4,3], dims=2)[:],"g-")
    legend()
    legend()
    xlabel("Time")
    ylabel("Stress")
    savefig("disp$i.png")
    savefig("disp$i.pdf")
    matwrite("nn$i.mat", Dict("U"=>U_, "S"=>Sigma_))
end

if mode=="data"
    U = run(sess, Us)
    S = run(sess, Ss)
    matwrite("invdata.mat", Dict("U"=>U, "S"=>S))
    global Sigma_ = S[6]
    global U_ = U[6]
    global Sigma0 = Sigma_
    global U0 = U_
    visualize("true")
    error("Stop!") 
end

# @show run(sess, loss)
# lineview(sess, pl, loss, [2.0;0.5;1.0], [5.0;5.0;5.0])
# savefig("line.png")

# gradview(sess, pl, loss, [5.0;5.0;5.0])
# savefig("line.png")

# meshview(sess, pl, loss, [2.0;0.5;1.0], 0.1, 0.1)
# savefig("line.png")

loss_ = BFGS!(sess, loss)
p = run(sess, pl)
matwrite("param.mat", Dict("loss"=>loss_, "p"=>p))