using Revise
using AdFem


n = 20
mmesh = Mesh(n, n, 1/n)
NT = 100
Δt = 1.0/NT 
xy = gauss_nodes(mmesh)
xg, yg = xy[:,1], xy[:,2]

xy = fem_nodes(mmesh)
x, y = xy[:,1], xy[:,2]
β = @. 1+xg^2+yg^2 + (1-xg)^2+(1-yg)^2 

f = (x, y, t)-> x*y*(x - 1)*(y - 1)*(-2*x^2 - 2*y^2 - 2*(x - 1)^2 - 2*(y - 1)^2 + (x^2 + y^2 + (x - 1)^2 + (y - 1)^2 + 1)^2 - 1)*exp(-t)
F = zeros(NT+1, get_ngauss(mmesh))

for i = 1:NT+1
    t = Δt * (i-1)
    F[i,:] = f.(xg, yg, t)
end


function r1_compute_rhs(u0, u1, f)
    compute_fem_source_term1(f, mmesh) + M0 * (2/Δt^2*u1 - u0/Δt^2) + M1 * u0 + M2 * u1 
end

function r1_solver(u0, u1, F, β)
    F = constant(F)
    β = constant(β)
    M0 = constant(compute_fem_mass_matrix1(mmesh))
    M1 = compute_fem_mass_matrix1(β/Δt, mmesh)
    M2 = compute_fem_mass_matrix1(-β^2, mmesh)
    M3 = compute_fem_mass_matrix1(1/Δt^2 + β/Δt, mmesh)
    RHS = zeros(mmesh.ndof)
    bdnode = bcnode(mmesh)
    M3, _ = impose_Dirichlet_boundary_conditions(M3, RHS, bdnode, zeros(length(bdnode)))


    function condition(i, uarr)
        return i <= NT
    end
    function body(i, uarr)
        u0 = read(uarr, i-1)
        u1 = read(uarr, i)
        RHS = compute_rhs(u0, u1, F[i])
        RHS = scatter_update(RHS, bdnode, zeros(length(bdnode)))
        u2 = M3\RHS
        i+1, write(uarr, i+1, u2)
    end

    uarr = TensorArray(NT+1)
    uarr = write(uarr, 1, u0)
    uarr = write(uarr, 2, u1)
    i = constant(2, dtype = Int32)
    _, uout = while_loop(condition, body, [i, uarr])
    u = stack(uout)
end

# prepare


exact_u = (x,y,t) -> x*(1-x)*y*(1-y)*exp(-t)
u0 = exact_u.(x, y, 0.0)
u1 = exact_u.(x, y, Δt)
u = r1_solver(u0, u1, F, β)

sess = Session();init(sess)
U = run(sess, u)

close("all")
figure(figsize=(15,4))
subplot(131)
visualize_scalar_on_fem_points(U[end,:], mmesh)
subplot(132)
visualize_scalar_on_fem_points(exact_u.(x, y, 1.0), mmesh)
subplot(133)
visualize_scalar_on_fem_points(abs.(U[end,:]-exact_u.(x, y, 1.0)), mmesh)
savefig("sol.png")