using Revise
using PyPlot 
using PoreFlow


function calc_residual_and_jacobian(uvnext, uv)
    unext, vnext = uvnext[1:mmesh.ndof], uvnext[mmesh.ndof+1:end]
    u, v = uv[1:mmesh.ndof], uv[mmesh.ndof+1:end]
    ugauss = eval_grad_on_gauss_pts1(unext, mmesh)
    vgauss = eval_grad_on_gauss_pts1(vnext, mmesh)
    ux, uy = ugauss[:,1], ugauss[:,2]
    vx, vy = vgauss[:,1], vgauss[:,2]
    ug = dof_to_gauss_points(unext, mmesh)
    vg = dof_to_gauss_points(vnext, mmesh)
    uo = dof_to_gauss_points(u, mmesh)
    vo = dof_to_gauss_points(v, mmesh)
    F = compute_fem_source_term((ug-uo)/Δt + ug * ux + vg * uy, (vg-vo)/Δt + ug * vx + vg * vy,  mmesh)
    G = [compute_fem_laplace_term1(unext, nu, mmesh); compute_fem_laplace_term1(vnext, nu, mmesh)]
    r = F - G

    F11 = compute_fem_mass_matrix1(1/Δt + ux, mmesh) + compute_fem_advection_matrix1(ug, vg, mmesh)
    F12 = compute_fem_mass_matrix1(uy, mmesh)
    F22 = compute_fem_mass_matrix1(1/Δt + vy, mmesh) + compute_fem_advection_matrix1(ug, vg, mmesh)
    F21 = compute_fem_mass_matrix1(vx, mmesh)
    DF = [F11 F12; F21 F22]

    DG = [compute_fem_laplace_matrix1(nu, mmesh) spzero(mmesh.ndof)
        spzero(mmesh.ndof) compute_fem_laplace_matrix1(nu, mmesh)]
    J = DF - DG 

    r, J 
end

function calc_next_step(uv)
    function f(θ, uvnext)
        r, J = calc_residual_and_jacobian(uvnext, uv)
        J, r = impose_Dirichlet_boundary_conditions(J, r, bdnode, zeros(length(bdnode)))
        r, J 
    end
    uvnext = newton_raphson_with_grad(f, uv, missing)
end

function solve_burgers(u0, NT)
    function condition(i, uarr)
        i <= NT 
    end
    function body(i, uarr)
        uv = read(uarr, i)
        uvnext = calc_next_step(uv)
        i+1, write(uarr, i+1, uvnext)
    end
    i = constant(1, dtype = Int32)
    uarr = TensorArray(NT+1)
    uarr = write(uarr, 1, u0)
    _, u = while_loop(condition, body, [i, uarr])
    set_shape(stack(u), NT+1, length(u0))
end

ADCME.options.newton_raphson.verbose = true
Δt = 0.01
mmesh = Mesh(30, 30, 1/30)
bdnode = bcnode(mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nu = constant(0.0001ones(get_ngauss(mmesh)))
nodes = fem_nodes(mmesh)
u = @. sin(2π * nodes[:,1])
v = @. cos(2π * nodes[:,2])
u0 = [u;v]
u0[bdnode] .= 0.0
us = solve_burgers(u0, 10)


sess = Session(); init(sess)
U = run(sess, us)
figure(figsize=(10,5))
close("all")
subplot(121)
visualize_scalar_on_fem_points(U[end,1:mmesh.nnode], mmesh)
subplot(122)
visualize_scalar_on_fem_points(U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("test_mfem.png")

close("all")
visualize_vector_on_fem_points(U[end,1:mmesh.nnode], U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("mfem_quiver.png")
# function ff(x)
#     unext = constant(x[1:mmesh.ndof])
#     vnext = constant(x[mmesh.ndof+1:end])
#     r, J = calc_residual_and_jacobian(unext, vnext, u, v, mmesh)
#     run(sess, r), run(sess, J)
# end

# ff(rand(2mmesh.ndof))
# sess = Session(); init(sess)
# test Jacobian 
# test_jacobian(ff, rand(2mmesh.ndof))