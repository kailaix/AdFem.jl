using Revise
using PyPlot 
using PoreFlow


function calc_residual_and_jacobian(unext, vnext, u, v, mmesh)
    ugauss = eval_grad_on_gauss_pts1(unext, mmesh)
    vgauss = eval_grad_on_gauss_pts1(vnext, mmesh)
    ux, uy = ugauss[:,1], ugauss[:,2]
    vx, vy = vgauss[:,1], vgauss[:,2]
    ug = dof_to_gauss_points(unext, mmesh)
    vg = dof_to_gauss_points(vnext, mmesh)
    uo = dof_to_gauss_points(u, mmesh)
    vo = dof_to_gauss_points(v, mmesh)
    F = compute_fem_source_term((ug-uo)/Δt + ug * ux + vg * uy, (vg-vo)/Δt + ug * vx + vg * vy,  mmesh)
    G = [compute_fem_laplace_term1(unext, ν, mmesh); compute_fem_laplace_term1(vnext, ν, mmesh)]
    r = F - G

    F11 = 1/Δt * spdiag(ones(mmesh.ndof)) + compute_fem_mass_matrix1(ux, mmesh) + compute_fem_advection_matrix1(ug, vg, mmesh)
    F12 = compute_fem_mass_matrix1(uy, mmesh)
    F22 = 1/Δt * spdiag(ones(mmesh.ndof)) + compute_fem_mass_matrix1(vy, mmesh) + compute_fem_advection_matrix1(ug, vg, mmesh)
    F21 = compute_fem_mass_matrix1(vx, mmesh)
    DF = [F11 F12; F21 F22]

    DG = [compute_fem_laplace_matrix1(ν, mmesh) spzero(mmesh.ndof)
        spzero(mmesh.ndof) compute_fem_laplace_matrix1(ν, mmesh)]
    J = DF - DG 

    r, J 
end