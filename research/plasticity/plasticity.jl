using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using ADCME
using MAT
using JLD2
using DelimitedFiles
using PyPlot
np = pyimport("numpy")
mpl = pyimport("tikzplotlib")


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


sess = Session(); init(sess)
A = run(sess, A)
M = run(sess, M)
L = run(sess, L)
H = run(sess, H)
Q = run(sess, Q)

# error()
x = Float64[]; y = Float64[]
for j = 1:n+1
    for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
    end
end
    
injection = (div(n,2)-1)*m + 3
production = (div(n,2)-1)*m + m-3


function get_disp(idx, ipval)
    local fint, stiff, α, σ
    U = zeros(m*n+2(m+1)*(n+1), NT+1)
    K, σY = 0.5, 0.1
    # K, σY = 0.5, 10000.0
    # K, σY = 0.5, 100.0
    σ0 = zeros(4*m*n, 3)
    α0 = zeros(4*m*n)
    Sigma = zeros(NT+1, 4*m*n, 3)
    for i = 1:NT 
        t = i*Δt
        @info i
            
        bdval = zeros(2*length(bdnode))
        up = copy(U[:, i])
        ε0 = eval_strain_on_gauss_pts(U[1:2(m+1)*(n+1),i], m, n, h)
        iter = 0
        while true
            iter += 1
            # @info size(ε0), size(σ0), size(α0)
            fint, stiff, α, σ = compute_planestressplasticity_stress_and_stiffness_matrix(
                up[1:2(m+1)*(n+1)], ε0, σ0, α0, K, σY, H, m, n, h
            )
            
            rhs1 = fint - b*L'*up[2(m+1)*(n+1)+1:end]
            rhs2 =  b*L/Δt*(up[1:2(m+1)*(n+1)] - U[1:2(m+1)*(n+1), i]) + 
                    M * (up[2(m+1)*(n+1)+1:end] - U[2(m+1)*(n+1)+1:end,i])/Δt - 
                    Q * up[2(m+1)*(n+1)+1:end] + Prhs
            rhs2[injection] -= ipval
            rhs2[production] += ipval
            rhs = [rhs1;rhs2]        
            rhs[[bdnode; bdnode.+ (m+1)*(n+1)]] = bdval 
            err = norm(rhs)
            @info err
            if err<1e-8
                break 
            end

            A = [stiff -b*L'
                b*L/Δt 1/Δt*M-Q]
            A, _ = fem_impose_coupled_Dirichlet_boundary_condition(A, bdnode, m, n, h)
            Δu = A\rhs
            up -= Δu
            
        end

        σ0, α0 = σ, α
        U[:,i+1] = up
        Sigma[i+1,:,:] = σ0
    end
    Dict(
        "U$idx"=>U, 
        "Sigma$idx"=>Sigma
    )
end

D = merge([get_disp(i, 0.2*i) for i = 1:5]...)
D = merge(D, get_disp(0, 0.5))


matwrite("plasticity.mat", D)
visualize_von_mises_stress(D["Sigma0"], m, n, h, name="_plasticity")
visualize_scattered_displacement(D["U0"], m, n, h, name="_plasticity", 
                xlim_=[-2h, m*h+2h], ylim_=[-2h, n*h+2h])