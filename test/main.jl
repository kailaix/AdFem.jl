
m = 20
n = 2m
h = 1/m
u = zeros(n+1, m+1)
v = zeros(n+1, m+1)
uxy = zeros(4, n+1, m+1)
bdnode = Int32[]
k = 0
Y = zeros(n, m)
for j = 1:n+1 
    for i = 1:m+1
        global k 
        k += 1
        x = (i-1)*h; y = (j-1)*h
        u[j, i] = x*(1-x)*y*(2-y)
        v[j, i] = x*(1-x)*y*(2-y)
        if i==1 || j==1 || i==m+1 || j==n+1
            push!(bdnode, k)
        end
    end
end
for j = 1:n
    for i = 1:m
        Y[j, i] = (j-1/2)*h
    end
end
# permi/Bf/μ*ρf
# Y

E = 1
ν = 0.3
k = E*(1-ν)/(1+ν)/(1-2ν) * [
    1 ν/(1-ν) ν/(1-ν)
    ν/(1-ν) 1 ν/(1-ν)
    ν/(1-ν) ν/(1-ν) 1
]

function step1(p, ρb)
    rhs = elem_average(zeros(n*m), ρb*g, m, n, h)
    uv = mechanics(k,rhs,bdnode, m, n, h)
    # e.g., in the case of neural network we have
    # uv = mechanics_nn(θ, rhs, bdnode, m, n, h)
    u, v = uv[1:(m+1)*(n+1)], uv[(m+1)*(n+1)+1:end]
    u, v 
end

function step2(u, v, u_, v_, p, b)
    veps = varepsilon(u, v, m, n, h) #volumetric strain
    veps_ = varepsilon(u_, v_, m, n, h) #volumetric strain
    v = b*(veps_-veps)
    t0 = p/M*h^2
    t1 = -cell_average(v, m, n, h)
    t2 = cell_average(f, m, n, h)
    t3 = tpfa(Δt * permi/Bf/μ*ρf * g * Y, m, n, h)
    p = flow_solver(M, Δt, t0+t1+t2+t3, m, n, h)
    p
end

# while true, do iteration:
# use custom_gradient to overload grad 
u, v = step1(p, ρb)
p = step2(u, v, u_, v_, p, b)

# stop when changes in p, u, v are sufficiently small