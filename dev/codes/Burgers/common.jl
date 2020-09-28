using Revise
using PyPlot 
using PoreFlow
using MAT


function calc_residual_and_jacobian(nu, uvnext, uv)
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

function calc_next_step(θ, uv)
    function f(θ, uvnext)
        r, J = calc_residual_and_jacobian(θ, uvnext, uv)
        J, r = impose_Dirichlet_boundary_conditions(J, r, bdnode, zeros(length(bdnode)))
        r, J 
    end
    uvnext = newton_raphson_with_grad(f, uv, θ)
end

"""
θ -- viscosity field (must be a vector of size ngauss)
NT -- total number of time steps 
u0 -- initial condition 
"""
function solve_burgers(u0, NT, θ)
    function condition(i, uarr)
        i <= NT 
    end
    function body(i, uarr)
        uv = read(uarr, i)
        uvnext = calc_next_step(θ, uv)
        i+1, write(uarr, i+1, uvnext)
    end
    i = constant(1, dtype = Int32)
    uarr = TensorArray(NT+1)
    uarr = write(uarr, 1, u0)
    _, u = while_loop(condition, body, [i, uarr])
    set_shape(stack(u), NT+1, length(u0))
end