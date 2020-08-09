using LinearAlgebra
using MAT
using PoreFlow
using PyPlot
using SparseArrays

# geometry setup in domain [0,1]^2
solid_left = 0.45
solid_right = 0.55
solid_top = 0.5
solid_bottom = 0.52

chip_left = 0.48
chip_right = 0.52
chip_top = 0.5
chip_bottom = 0.505

k_mold = 0.014531
k_chip = 2.60475
k_air = 0.64357
nu = 0.47893 # equal to 1/Re
power_source = 82.46295 #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

m = 200
n = 200
h = 1/n
NT = 5    # number of iterations for Newton's method, 8 is good for m=400


# compute solid indices and chip indices
solid_fem_idx = Array{Int64, 1}([])
solid_fvm_idx = Array{Int64, 1}([])
chip_fem_idx = Array{Int64, 1}([])
chip_fvm_idx = Array{Int64, 1}([])

for i = 1:(m+1)
    for j = 1:(n+1)
        if (i-1)*h >= solid_left-1e-9 && (i-1)*h <= solid_right+1e-9 && (j-1)*h >= solid_top-1e-9 && (j-1)*h <= solid_bottom+1e-9
            # print(i, j)
            global solid_fem_idx = [solid_fem_idx; j*(m+1)+i]
            if (i-1)*h >= chip_left-1e-9 && (i-1)*h <= chip_right+1e-9 && (j-1)*h >= chip_top-1e-9 && (j-1)*h <= chip_bottom+1e-9
                global chip_fem_idx = [chip_fem_idx; j*(m+1)+i]
            end
        end
    end
end

for i = 1:m
    for j = 1:n
        if (i-1)*h + h/2 >= solid_left-1e-9 && (i-1)*h + h/2 <= solid_right+1e-9 && 
            (j-1)*h + h/2 >= solid_top-1e-9 && (j-1)*h + h/2 <= solid_bottom+1e-9
            global solid_fvm_idx = [solid_fvm_idx; j*m+i]
            if (i-1)*h + h/2 >= chip_left-1e-9 && (i-1)*h + h/2 <= chip_right+1e-9 && (j-1)*h + h/2 >= chip_top-1e-9 && (j-1)*h + h/2<= chip_bottom+1e-9
                global chip_fvm_idx = [chip_fvm_idx; j*m+i]
            end
        end
    end
end

# initialize space varying k and heat source
# k_fem = k_air * ones((m+1)*(n+1))
# k_fem[solid_fem_idx] .= k_mold
# k_fem[chip_fem_idx] .= k_chip 

k_fem = k_air * constant(ones((m+1)*(n+1)))
k_fem = scatter_update(k_fem, solid_fem_idx, k_mold * ones(length(solid_fem_idx)))
k_fem = scatter_update(k_fem, chip_fem_idx, k_chip * ones(length(chip_fem_idx)))
kgauss = fem_to_gauss_points(k_fem, m, n, h)

heat_source_fem = zeros((m+1)*(n+1))
heat_source_fem[chip_fem_idx] .= power_source #/ h^2
heat_source_gauss = fem_to_gauss_points(heat_source_fem, m, n, h)

# chip_gauss_idx = [ 4 .* chip_fvm_idx; 4 .* chip_fvm_idx .- 1; 4 .* chip_fvm_idx .- 2; 4 .* chip_fvm_idx .- 3]
# heat_source_gauss = zeros(4*m*n)
# heat_source_gauss[chip_gauss_idx] .= power_source

# F1 = compute_fem_source_term1(eval_f_on_gauss_pts(ffunc_, m, n, h), m, n, h)
# F2 = compute_fem_source_term1(eval_f_on_gauss_pts(gfunc_, m, n, h), m, n, h)
# H = h^2*eval_f_on_fvm_pts(hfunc_, m, n, h)
B = constant(compute_interaction_matrix(m, n, h))

# compute F
Laplace = nu * constant(compute_fem_laplace_matrix1(m, n, h))
heat_source = constant(compute_fem_source_term1(heat_source_gauss, m, n, h))

LaplaceK = constant(compute_fem_laplace_matrix1(kgauss, m, n, h))
# xy = fem_nodes(m, n, h)
# x, y = xy[:, 1], xy[:, 2]
# k = @. k_nn(x, y); k=stack(k)
# kgauss = fem_to_gauss_points(k, m, n, h)
# LaplaceK = compute_fem_laplace_matrix1(kgauss, m, n, h)

bd = bcnode("all", m, n, h)

# only apply Dirichlet to velocity; set left bottom two points to zero to fix rank deficient problem for pressure

bd = [bd; bd .+ (m+1)*(n+1); 
     2*(m+1)*(n+1)+1; 2*(m+1)*(n+1)+m;
    #  (2*(m+1)*(n+1)+m*n )+1:(2*(m+1)*(n+1)+m*n )+m+1]
     bd .+ (2*(m+1)*(n+1)+m*n )]

# add solid region into boundary condition for u, v, p
bd = [bd; solid_fem_idx; solid_fem_idx .+ (m+1)*(n+1); solid_fvm_idx .+ 2(m+1)*(n+1)]


function compute_residual(S)
    u, v, p, T = S[1:(m+1)*(n+1)], 
        S[(m+1)*(n+1)+1:2(m+1)*(n+1)], 
        S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n],
        S[2(m+1)*(n+1)+m*n+1:end]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    interaction = compute_interaction_term(p, m, n, h) # julia kernel needed
    f1 = compute_fem_source_term1(ugauss.*ux, m, n, h)
    f2 = compute_fem_source_term1(vgauss.*uy, m, n, h)
    f3 = -interaction[1:(m+1)*(n+1)]
    f4 = Laplace*u 
    # f5 = -F1
    F = f1 + f2 + f3 + f4 #+ f5 

    g1 = compute_fem_source_term1(ugauss.*vx, m, n, h)
    g2 = compute_fem_source_term1(vgauss.*vy, m, n, h)
    g3 = -interaction[(m+1)*(n+1)+1:end]    
    g4 = Laplace*v 
    # g5 = -F2
    T_gauss = fem_to_gauss_points(T, m, n, h)
    buoyance_term = - buoyance_coef * compute_fem_source_term1(T_gauss, m, n, h)

    G = g1 + g2 + g3 + g4 + buoyance_term #+ g5

    H0 = -B * [u;v] # + H

    T0 = LaplaceK * T + compute_fem_advection_matrix1(ugauss,vgauss, m, n, h) * T - heat_source
    R = [F;G;H0;T0]
    return R
end

function compute_jacobian(S)
    u, v, p, T = S[1:(m+1)*(n+1)], 
        S[(m+1)*(n+1)+1:2(m+1)*(n+1)], 
        S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n],
        S[2(m+1)*(n+1)+m*n+1:end]
        
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    M1 = constant(compute_fem_mass_matrix1(ux, m, n, h))
    M2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h)) # a julia kernel needed
    M3 = Laplace
    Fu = M1 + M2 + M3 

    Fv = constant(compute_fem_mass_matrix1(uy, m, n, h))

    N1 = constant(compute_fem_mass_matrix1(vy, m, n, h))
    N2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h))
    N3 = Laplace
    Gv = N1 + N2 + N3 

    Gu = constant(compute_fem_mass_matrix1(vx, m, n, h))

    M = LaplaceK + constant(compute_fem_advection_matrix1(ugauss,vgauss, m, n, h))

    gradT = eval_grad_on_gauss_pts1(T, m, n, h)
    Tx, Ty = gradT[:,1], gradT[:,2]
    DU_TX = constant(compute_fem_mass_matrix1(Tx, m, n, h))       # (m+1)*(n+1), (m+1)*(n+1)
    DV_TY = constant(compute_fem_mass_matrix1(Ty, m, n, h))       # (m+1)*(n+1), (m+1)*(n+1)

    T_mat = -buoyance_coef * constant(compute_fem_mass_matrix1(m, n, h))
    T_mat = [SparseTensor(spzeros((m+1)*(n+1), (m+1)*(n+1))); T_mat]

    J0 = [Fu Fv
          Gu Gv]

    J1 = [J0 -B' T_mat
        -B spdiag(zeros(size(B,1))) SparseTensor(spzeros(m*n, (m+1)*(n+1)))]
    
    N = 2*(m+1)*(n+1) + m*n 
    J = [J1 
        [DU_TX DV_TY SparseTensor(spzeros((m+1)*(n+1), m*n)) M]]
end

function solve_steady_cavityflow_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, m, n, h)
    residual = scatter_update(residual, bd, zeros(length(bd)))    # residual[bd] .= 0.0 in Tensorflow syntax

    d = J\residual
    residual_norm = norm(residual)
    op = tf.print("residual norm", residual_norm)
    d = bind(d, op)
    S_new = S - d
    return S_new
end


function condition(i, S_arr)
    i <= NT + 1
end

function body(i, S_arr)
    S = read(S_arr, i-1)
    op = tf.print("i=",i)
    i = bind(i, op)
    S_new = solve_steady_cavityflow_one_step(S)
    S_arr = write(S_arr, i, S_new)
    return i+1, S_arr
end

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, zeros(m*n+3*(m+1)*(n+1)))

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, 2*(m+1)*(n+1)+m*n+(m+1)*(n+1)))

sess = Session(); init(sess)
output = run(sess, S)


matwrite("xzchipv0_data.mat", 
    Dict(
        "V"=>output[end, :]
    ))


u0, v0, p0, t0 = zeros((m+1)*(n+1)), zeros((m+1)*(n+1)), zeros(m*n), zeros((m+1)*(n+1))



figure(figsize=(10,10))
subplot(221)
title("u velocity")
visualize_scalar_on_fem_points(output[NT+1, 1:(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
subplot(222)
title("v velocity")
visualize_scalar_on_fem_points(output[NT+1, (m+1)*(n+1)+1:2*(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
subplot(223)
visualize_scalar_on_fvm_points(output[NT+1, 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std, m, n, h);gca().invert_yaxis()
title("pressure")
subplot(224)
title("temperature")
visualize_scalar_on_fem_points(output[NT+1, 2*(m+1)*(n+1)+m*n+1:end].* T_infty .+ T_infty, m, n, h);gca().invert_yaxis()
tight_layout()

print("Solution range:",
    "\n [u velocity] \t min:", minimum(output[NT+1, 1:(m+1)*(n+1)] .* u_std), ",\t max:", maximum(output[NT+1, 1:(m+1)*(n+1)] .* u_std),
    "\n [v velocity] \t min:", minimum(output[NT+1, (m+1)*(n+1)+1:2*(m+1)*(n+1)] .* u_std), ",\t max:", maximum(output[NT+1, (m+1)*(n+1)+1:2*(m+1)*(n+1)] .* u_std),
    "\n [pressure]   \t min:", minimum(output[NT+1, 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std), ",\t max:", maximum(output[NT+1, 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std),
    "\n [temperature]\t min:", minimum(output[NT+1, 2*(m+1)*(n+1)+m*n+1:end].* T_infty .+ T_infty), ",\t\t\t max:", maximum(output[NT+1, 2*(m+1)*(n+1)+m*n+1:end].* T_infty .+ T_infty))


# separate plots
# figure();visualize_scalar_on_fem_points(output[NT+1, 1:(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
# figure();visualize_scalar_on_fem_points(output[NT+1, (m+1)*(n+1)+1:2*(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
# figure();visualize_scalar_on_fvm_points(output[NT+1, 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std, m, n, h);gca().invert_yaxis()
# figure();visualize_scalar_on_fem_points(output[NT+1, 2*(m+1)*(n+1)+m*n+1:end].* T_infty .+ T_infty, m, n, h);gca().invert_yaxis()
