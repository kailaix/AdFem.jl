using Revise
using AdFem
using PyPlot

function ricker(;dt = 0.005, f0 = 3.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    w = @. (1 - 2b)*exp(-b)
end

n = 50
mmesh = Mesh(n, n, 1/n)
NT = 300
Δt = 6/NT 

xy = gauss_nodes(mmesh)
xg, yg = xy[:,1], xy[:,2]
xy = fem_nodes(mmesh)
x, y = xy[:,1], xy[:,2]

β = constant(zeros(mmesh.ndof))
β_g = constant(zeros(get_ngauss(mmesh)))
βprime = constant(zeros(get_ngauss(mmesh)))
c = constant(0.1ones(get_ngauss(mmesh)))
nv = [ones(get_ngauss(mmesh)) zeros(get_ngauss(mmesh))]
nv = constant(nv)
F = zeros(NT+1, get_ngauss(mmesh))
F[1:length(ricker(f0=7.0)),2312] = ricker(f0=7.0)
F = constant(F)

RHS = zeros(mmesh.ndof)
bdnode = bcnode(mmesh)

M = constant(compute_fem_mass_matrix1(mmesh))
Z1 = constant(compute_fem_mass_matrix1(β_g, mmesh))
Z2 = constant(compute_fem_mass_matrix1(β_g^2, mmesh))
M, _ = impose_Dirichlet_boundary_conditions(M, RHS, bdnode, zeros(length(bdnode)))
Z1, _ = impose_Dirichlet_boundary_conditions(Z1, RHS, bdnode, zeros(length(bdnode)))
Z2, _ = impose_Dirichlet_boundary_conditions(Z2, RHS, bdnode, zeros(length(bdnode)))

integrator1 = ExplicitNewmark(M, 2Z1, Z2, Δt)
integrator2 = integrator1 
integrator3 = ExplicitNewmark(M, Z1, missing, Δt)
integrator4 = ExplicitNewmark(M, missing, missing, Δt)

function rk_one_step(d2, t)
    h1 = -β*d2 + t 
    h2 = -β*(d2+Δt*h1)+t 
    d2 + Δt/2*(h1+h2)
end

function one_step(u, d10, d20, d30, d40, t0,
                    d11, d21, d31, d41, t1, f)
    k1, k2, k3, k4 = compute_pml_term(u, βprime, c, nv, mmesh)
    k4 = k4 + compute_fem_source_term1(f, mmesh)
    d12 = step(integrator1, d10, d11, k1)
    t2 = step(integrator2, t0, t1, k2)
    d32 = step(integrator3, d30, d31, k3)
    d42 = step(integrator4, d40, d41, k4)
    d22 = rk_one_step(d21, t2)
    u = d12 + d22 + d32 + d42
    return u, d12, d22, d32, d42, t2 
end

function simulate()
    for op in [:d1, :d2, :d3, :d4, :t, :uarr]
        @eval $op=TensorArray(NT+1)
        @eval $op=write($op, 1, zeros(mmesh.ndof))
        @eval $op=write($op, 2, zeros(mmesh.ndof))
    end
    
    function condition(i, tas...)
        i<=NT
    end
    function body(i, d1, d2, d3, d4, t, uarr)
        d10 = read(d1, i-1); d11 = read(d1, i)
        d20 = read(d2, i-1); d21 = read(d2, i)
        d30 = read(d3, i-1); d31 = read(d3, i)
        d40 = read(d4, i-1); d41 = read(d4, i)
        t0 = read(t, i-1); t1 = read(t, i)
        u = read(uarr, i)
        f = F[i]
        u, d12, d22, d32, d42, t2  = one_step(u, d10, d20, d30, d40, t0,
                    d11, d21, d31, d41, t1, f)
        i+1, write(d1, i+1, d12), write(d2, i+1, d22), write(d3, i+1, d32), 
                write(d4, i+1, d42), write(t, i+1, t2), write(uarr, i+1, u)
    end
    i = constant(2, dtype = Int32)
    _, _, _, _, _, _, u = while_loop(condition, body, [i, d1, d2, d3, d4, t, uarr])
    set_shape(stack(u), (NT+1, mmesh.ndof))
end


u = simulate()
sess = Session(); init(sess)
U = run(sess, u)

vmax, vmin = maximum(U), minimum(U)
for k = 0:5
    close("all")
    visualize_scalar_on_fem_points(U[k * (NT÷6) + 1,:], mmesh, vmin = vmin, vmax = vmax)
    savefig("s$k.png")
end