using Revise
using AdFem
using PyCall
using LinearAlgebra
using ADCME
using MAT
using JLD2
using PyPlot
np = pyimport("numpy")

# Domain information 
NT = 20
Δt = 1/NT
n = 10
m = 2*n 
h = 1.0/n 
bdnode = bcnode("lower", m, n, h)



b = 1.0
λ = constant(2.0)
μ = constant(0.5)
invη = constant(1.0)


H_ = zeros(4*m*n, 3, 3)
for i = 1:4*m*n 
    H_[i,:,:] = diagm(0=>ones(3))
end
H_ = Variable(H_)
H = map(x->x'*x, H_)


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

        g = -batch_matmul(H, ε0)
        rhs1 = compute_strain_energy_term(g, m, n, h)
        
        rhs1 = scatter_update(rhs1, [bdnode; bdnode .+ (m+1)*(n+1)], zeros(2length(bdnode)))
        rhs2 = zeros(m*n)
        rhs2[injection] += 1.0 
        rhs2[production] -= 1.0
        rhs2 = rhs2 * ipval + b*L*u[1:2(m+1)*(n+1)]/Δt + 
                M * u[2(m+1)*(n+1)+1:end]/Δt + Prhs
        
        rhs = [rhs1;rhs2]
        o = A\rhs 

        ε = eval_strain_on_gauss_pts(o, m, n, h)
        σ = batch_matmul(H, ε)
        
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

d = matread("invdata.mat")
Us_ = d["U"]
Ss_ = d["S"]
Sigma0 = Ss_[6]
U0 = Us_[6]
loss = sum([sum((Us[i][:,1:m+1] - Us_[i][:,1:m+1])^2) for i = 1:5])
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

# run(sess, loss)
# gradview(sess, H_, loss, diagm(0=>ones(3))[:])
# savefig("line.png")

loss = loss*1e10
U_, Sigma_ = run(sess, [Us[6], Ss[6]])
visualize(0)
for iter = 1:5000
    BFGS!(sess, loss, 500)
    global U_, Sigma_ = run(sess, [Us[6], Ss[6]])
    visualize(iter)
end
