using Revise
using AdFem
using PyCall
using LinearAlgebra
np = pyimport("numpy")

# Domain information 
NT = 50
Δt = 1/NT
n = 50
m = n 
h = 1.0/n 
bdnode = Int64[]
for i = 1:m+1
    for j = 1:n+1
        if i==1 || i==m+1 || j==1|| j==n+1
            push!(bdnode, (j-1)*(m+1)+i)
        end
    end
end

# Physical parameters
b = 1.0
H = [1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 0.5]
Q = compute_fvm_tpfa_matrix(m, n, h)
K = compute_fem_stiffness_matrix(H, m, n, h)
L = compute_interaction_matrix(m, n, h)
M = compute_fvm_mass_matrix(m, n, h)
A = [K -b*L'
b*L/Δt 1/Δt*M-Q]
A, Abd = fem_impose_coupled_Dirichlet_boundary_condition(A, bdnode, m, n, h)

pexact = (x,y,t)->x^2*(1-x)^2*y^2*(1-y)^2*exp(-t)
uexact1 = (x,y,t)->(x^2+y^2)*t
uexact2 = (x,y,t)->(x^2-y^2)*t
g1 = (x,y,t)-> 3t + (-x^2*y^2*(1 - y)^2*(2*x - 2)*exp(-t) - 2*x*y^2*(1 - x)^2*(1 - y)^2*exp(-t))
g2 = (x,y,t)-> -t + (-x^2*y^2*(1 - x)^2*(2*y - 2)*exp(-t) - 2*x^2*y*(1 - x)^2*(1 - y)^2*exp(-t))
f = (x,y,t)->(-x^2*y^2*(x - 1)^2*(y - 1)^2 - 2*x^2*y^2*(x - 1)^2 - 2*x^2*y^2*(y - 1)^2 - 8*x^2*y*(x - 1)^2*(y - 1) - 2*x^2*(x - 1)^2*(y - 1)^2 - 8*x*y^2*(x - 1)*(y - 1)^2 - 2*y^2*(x - 1)^2*(y - 1)^2 + 2*(x - y)*exp(t))*exp(-t)

U = zeros(m*n+2(m+1)*(n+1), NT+1)
for i = 1:m 
    for j = 1:n 
        x = (i-1/2)*h; y = (j-1/2)*h 
        U[2(m+1)*(n+1)+(j-1)*m+i, 1] = pexact(x,y,0.0)
    end
end

x = Float64[]; y = Float64[]
for j = 1:n+1
    for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
    end
end
    
for i = 1:NT 
    t = i*Δt
    @info i
    g1_ = eval_f_on_gauss_pts((x,y)->g1(x,y,t), m, n, h)
    g2_ = eval_f_on_gauss_pts((x,y)->g2(x,y,t), m, n, h)
    f_ = eval_f_on_gauss_pts((x,y)->f(x,y,t), m, n, h)
    rhs1 = -compute_fem_source_term(g1_, g2_, m, n, h)
    rhs2 = -compute_fvm_source_term(f_, m, n, h) + 
                b*L*U[1:2(m+1)*(n+1), i]/Δt + 
                M * U[2(m+1)*(n+1)+1:end,i]/Δt

    rhs = [rhs1;rhs2]

    bdval = [uexact1.(x[bdnode],y[bdnode],t); uexact2.(x[bdnode],y[bdnode],t)]
    rhs[[bdnode; bdnode.+ (m+1)*(n+1)]] = bdval 
    rhs -= Abd * bdval 

    U[:,i+1] = A\rhs
end

U0 = zeros(2*(m+1)*(n+1)+m*n, NT+1)
for tk = 1:NT+1
    t = (tk-1)*Δt
    for j = 1:n+1
        for i = 1:m+1
            x = (i-1)*h 
            y = (j-1)*h 
            U0[(j-1)*(m+1)+i, tk] = uexact1(x, y, t)
            U0[(j-1)*(m+1)+i+(m+1)*(n+1), tk] = uexact2(x, y, t)
        end
    end
    for i = 1:m 
        for j = 1:n 
            x = (i-1/2)*h 
            y = (j-1/2)*h
            U0[2*(m+1)*(n+1)+(j-1)*m+i, tk] = pexact(x, y, t)
        end
    end 
end

visualize_displacement(U, m, n, h, name="_out")
visualize_pressure(U, m, n, h, name="_out")

visualize_displacement(abs.(U-U0), m, n, h; name="_diff")
visualize_pressure(abs.(U-U0), m, n, h; name="_diff")

