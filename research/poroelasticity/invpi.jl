using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using ADCME
using MAT
using PyPlot
np = pyimport("numpy")

# Domain information 
NT = 50
Δt = 1/NT
n = 20
m = 2n 
h = 1.0/n 
bdnode = Int64[]
for i = 1:m+1
    for j = 1:n+1
        if j==n+1
            push!(bdnode, (j-1)*(m+1)+i)
        end
    end
end

is_training = false
b = 1.0
# Physical parameters
if !is_training
    E = 1.0
    ν = 0.35
    global H = E/(1-ν^2)*[1.0 ν 0
        ν 1 0
        0 0 1-ν]|>constant
else
    H = diagm(0=>ones(3))
    global H = Variable(H)
end
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

function condition(i, ta_u)
    i<=NT
end

function body(i, ta_u)
    u = read(ta_u, i)
    bdval = zeros(2*length(bdnode))
    rhs1 = vector([bdnode; bdnode.+ (m+1)*(n+1)], bdval, 2(m+1)*(n+1))
    rhs2 = zeros(m*n)
    rhs2[injection] += 1.0
    rhs2[production] -= 1.0
    rhs2 += b*L*u[1:2(m+1)*(n+1)]/Δt + 
            M * u[2(m+1)*(n+1)+1:end]/Δt
    rhs = [rhs1;rhs2]
    rhs -= Abd * bdval 
    o = A\rhs 
    ta_u = write(ta_u, i+1, o)
    i+1, ta_u
end

i = constant(1, dtype=Int32)
ta_u = TensorArray(NT+1)
ta_u = write(ta_u, 1, constant(zeros(2(m+1)*(n+1)+m*n)))
_, u_out = while_loop(condition, body, [i, ta_u]; parallel_iterations=1)
u_out = stack(u_out)
u_out.set_shape((NT+1, size(u_out,2)))


# upper_idx = Int64[]
# for i = 1:m+1
#     push!(upper_idx, (div(n,3)-1)*(m+1)+i)
#     push!(upper_idx, (div(n,3)-1)*(m+1)+i + (m+1)*(n+1))
# end
# for i = 1:m 
#     push!(upper_idx, (div(n,3)-1)*m+i+2(m+1)*(n+1))
# end
upper_idx = collect(1:m+1)

if is_training
    Ue = matread("U.mat")["U"]
    loss = sum((u_out[:, upper_idx] - Ue)^2)
    sess = Session(); init(sess)
    loss_ = BFGS!(sess, loss, 200)

    close("all")
    semilogy(loss_)
    xlabel("Iterations")
    ylabel("Loss")
    grid("both")
    savefig("loss.png")
else
    sess = Session(); init(sess)
    U = run(sess, u_out)
    matwrite("U.mat", Dict("U"=> U[:, upper_idx]))
    visualize_displacement(U'|>Array, m, n, h, name="_tf")
    visualize_pressure(U'|>Array, m, n, h, name="_tf")

end

