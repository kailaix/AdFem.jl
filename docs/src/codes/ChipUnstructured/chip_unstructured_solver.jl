using LinearAlgebra
using MAT
using PoreFlow
using PyPlot; matplotlib.use("agg")
using SparseArrays

ADCME.options.sparse.auto_reorder = false

function compute_residual_and_jacobian(k_chip, S)
    # compute r and J in Jx=r for Newton's method
    # read in current step 
    k_fem = k_air * constant(ones(ndof))
    k_fem = scatter_update(k_fem, solid_fem_idx, k_mold * ones(length(solid_fem_idx)))
    # k_fem = scatter_update(k_fem, chip_fem_idx, k_chip * ones(length(chip_fem_idx))) # if k_chip is a constant
    k_fem = scatter_update(k_fem, chip_fem_idx, k_chip)
    kgauss = dof_to_gauss_points(k_fem, mesh)
    LaplaceK = constant(compute_fem_laplace_matrix1(kgauss, mesh))

    u, v, p, T = S[1:ndof], 
        S[ndof+1:2*ndof], 
        S[2*ndof+1:2*ndof+nelem],
        S[2*ndof+nelem+1:end]
        
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)

    ugauss = dof_to_gauss_points(u, mesh)
    vgauss = dof_to_gauss_points(v, mesh)
    ux, uy, vx, vy = grad_u[:,1], grad_u[:,2], grad_v[:,1], grad_v[:,2]

    # comupute jacobian
    M1 = constant(compute_fem_mass_matrix1(ux, mesh))
    M2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), mesh)) # a julia kernel needed
    M3 = Laplace
    Fu = M1 + M2 + M3

    Fv = constant(compute_fem_mass_matrix1(uy, mesh))

    N1 = constant(compute_fem_mass_matrix1(vy, mesh))
    N2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), mesh))
    N3 = Laplace
    Gv = N1 + N2 + N3

    Gu = constant(compute_fem_mass_matrix1(vx, mesh))

    M = LaplaceK + constant(compute_fem_advection_matrix1(ugauss,vgauss, mesh))

    gradT = eval_grad_on_gauss_pts1(T, mesh)
    Tx, Ty = gradT[:,1], gradT[:,2]
    DU_TX = constant(compute_fem_mass_matrix1(Tx, mesh))
    DV_TY = constant(compute_fem_mass_matrix1(Ty, mesh))

    T_mat = constant(compute_fem_mass_matrix1(-buoyance_coef * constant(ones(ndof)), mesh))
    T_mat = [SparseTensor(spzeros(ndof, ndof)); T_mat]

    J0 = [Fu Fv
          Gu Gv]

    J1 = [J0 -B' T_mat
        -B spdiag(zeros(size(B,1))) SparseTensor(spzeros(nelem, ndof))]
    
    J = [J1 
        [DU_TX DV_TY SparseTensor(spzeros(ndof, nelem)) M]]

    # compute residual
    interaction = compute_interaction_term(p, mesh)
    f1 = compute_fem_source_term1(ugauss.*ux, mesh)
    f2 = compute_fem_source_term1(vgauss.*uy, mesh)
    f3 = -interaction[1:ndof]
    f4 = Laplace*u 
    F = f1 + f2 + f3 + f4

    g1 = compute_fem_source_term1(ugauss.*vx, mesh)
    g2 = compute_fem_source_term1(vgauss.*vy, mesh)
    g3 = -interaction[ndof+1:end]    
    g4 = Laplace*v 
    T_gauss = dof_to_gauss_points(T, mesh)
    buoyance_term = - buoyance_coef * compute_fem_source_term1(T_gauss, mesh)
    G = g1 + g2 + g3 + g4 + buoyance_term

    H0 = -B * [u;v]

    T0 = LaplaceK * T + compute_fem_advection_matrix1(ugauss,vgauss, mesh) * T - heat_source

    R = [F;G;H0;T0]

    return R, J

end

function solve_one_step(θ, S)
    function f(θ, S)
        r, J = compute_residual_and_jacobian(θ, S)
        J, r = impose_Dirichlet_boundary_conditions(J, r, bd, zeros(length(bd)))
        # residual_norm = norm(r)
        # op = tf.print("residual norm", residual_norm)
        # r = bind(r, op)
        return r, J
    end

    S_new = newton_raphson_with_grad(f, S, θ)
    return S_new
end

function solve_navier_stokes(S0, NT, θ)
    function condition(i, S_arr)
        i <= NT
    end

    function body(i, S_arr)
        S = read(S_arr, i)
        op = tf.print("i=",i)
        i = bind(i, op)
        S_new = solve_one_step(θ, S)
        S_arr = write(S_arr, i+1, S_new)
        return i+1, S_arr
    end

    i = constant(1, dtype=Int32)
    S_arr = TensorArray(NT+1)
    S_arr = write(S_arr, 1, S0)
    _, S = while_loop(condition, body, [i, S_arr])
    S = set_shape(stack(S), NT+1, length(S0))
end
