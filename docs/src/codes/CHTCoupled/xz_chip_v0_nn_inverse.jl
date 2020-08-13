using LinearAlgebra
using MAT
using PoreFlow
using PyPlot
using SparseArrays

function k_exact(x, y)
    3.0 + 100000 * (x - 0.5)^3 / (1 + y^2)
    # 5 * exp((-(x-0.5)^2-(y-0.5)^2)/ 0.00002) + 
    # 3 * exp((-(x-0.48)^2-(y-0.505)^2)/ 0.00005) + 
    # 6 * exp((-(x-0.51)^2-(y-0.502)^2)/ 0.00001) + 2.604
end

function k_nn(x, y)
    out = fc([x y], [20,20,20,1])^2 + 0.5 # N x 1 
    squeeze(out)
end

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
k_air = 0.64357
nu = 0.47893 # equal to 1/Re
power_source = 0.06189 #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

m = 200
n = 200
h = 1/n
NT = 7    # number of iterations for Newton's method, 8 is good for m=400


# compute solid indices and chip indices
solid_fem_idx = Array{Int64, 1}([])
solid_fvm_idx = Array{Int64, 1}([])
chip_fem_idx = Array{Int64, 1}([])
chip_fvm_idx = Array{Int64, 1}([])
chip_fem_top_idx = Array{Int64, 1}([])

for i = 1:(m+1)
    for j = 1:(n+1)
        if (i-1)*h >= solid_left-1e-9 && (i-1)*h <= solid_right+1e-9 && (j-1)*h >= solid_top-1e-9 && (j-1)*h <= solid_bottom+1e-9
            # print(i, j)
            global solid_fem_idx = [solid_fem_idx; (j-1)*(m+1)+i]
            if (i-1)*h >= chip_left-1e-9 && (i-1)*h <= chip_right+1e-9 && (j-1)*h >= chip_top-1e-9 && (j-1)*h <= chip_bottom+1e-9
                global chip_fem_idx = [chip_fem_idx; (j-1)*(m+1)+i]
            end
            if (i-1)*h >= chip_left-1e-9 && (i-1)*h <= chip_right+1e-9 && (j-1)*h >= chip_top-1e-9 && (j-1)*h <= chip_top+1e-9
                global chip_fem_top_idx = [chip_fem_top_idx; (j-1)*(m+1)+i]
            end
        end
    end
end

for i = 1:m
    for j = 1:n
        if (i-1)*h + h/2 >= solid_left-1e-9 && (i-1)*h + h/2 <= solid_right+1e-9 && 
            (j-1)*h + h/2 >= solid_top-1e-9 && (j-1)*h + h/2 <= solid_bottom+1e-9
            global solid_fvm_idx = [solid_fvm_idx; (j-1)*m+i]
            if (i-1)*h + h/2 >= chip_left-1e-9 && (i-1)*h + h/2 <= chip_right+1e-9 && (j-1)*h + h/2 >= chip_top-1e-9 && (j-1)*h + h/2<= chip_bottom+1e-9
                global chip_fvm_idx = [chip_fvm_idx; (j-1)*m+i]
            end
        end
    end
end

# initialize space varying k and heat source

xy = fem_nodes(m, n, h)
chip_x, chip_y = xy[chip_fem_idx, 1], xy[chip_fem_idx, 2]
k_chip = @. k_nn(chip_x, chip_y); k_chip=stack(k_chip)
k_chip_exact = @. k_exact(chip_x, chip_y)

k_fem = k_air * constant(ones((m+1)*(n+1)))
k_fem = scatter_update(k_fem, solid_fem_idx, k_mold * ones(length(solid_fem_idx)))
k_fem = scatter_update(k_fem, chip_fem_idx, k_chip)
kgauss = fem_to_gauss_points(k_fem, m, n, h)

heat_source_fem = zeros((m+1)*(n+1))
heat_source_fem[chip_fem_idx] .= power_source #/ h^2
heat_source_fem[chip_fem_top_idx] .= 82.46295
heat_source_gauss = fem_to_gauss_points(heat_source_fem, m, n, h)

# chip_gauss_idx = [ 4 .* chip_fvm_idx; 4 .* chip_fvm_idx .- 1; 4 .* chip_fvm_idx .- 2; 4 .* chip_fvm_idx .- 3]
# heat_source_gauss = zeros(4*m*n)
# heat_source_gauss[chip_gauss_idx] .= power_source

B = constant(compute_interaction_matrix(m, n, h))

# compute F
Laplace = nu * constant(compute_fem_laplace_matrix1(m, n, h))
heat_source = constant(compute_fem_source_term1(heat_source_gauss, m, n, h))

LaplaceK = constant(compute_fem_laplace_matrix1(kgauss, m, n, h))

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

function plot_velo_pres_temp_cond(k) 

    S_true = V_data
    S = run(sess, V_computed)

    figure(figsize=(15,4))
    subplot(131)
    title("exact x velocity")
    visualize_scalar_on_fem_points(S_true[1:(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
    subplot(132)
    title("predicted x velocity")
    visualize_scalar_on_fem_points(S[1:(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
    subplot(133)
    title("difference in x velocity")
    visualize_scalar_on_fem_points(S[1:(m+1)*(n+1)] .* u_std .- S_true[1:(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
    tight_layout()
    savefig("xzchip_figures1/xzchipv0_nn_velox$k.png")

    figure(figsize=(15,4))
    subplot(131)
    title("exact y velocity")
    visualize_scalar_on_fem_points(S_true[(m+1)*(n+1)+1: 2*(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
    subplot(132)
    title("predicted y velocity")
    visualize_scalar_on_fem_points(S[(m+1)*(n+1)+1: 2*(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
    subplot(133)
    title("difference in y velocity")
    visualize_scalar_on_fem_points(S[(m+1)*(n+1)+1: 2*(m+1)*(n+1)]  .* u_std .- S_true[(m+1)*(n+1)+1: 2*(m+1)*(n+1)] .* u_std, m, n, h);gca().invert_yaxis()
    tight_layout()
    savefig("xzchip_figures1/xzchipv0_nn_veloy$k.png")


    figure(figsize=(15,4))
    subplot(131)
    title("exact pressure")
    visualize_scalar_on_fvm_points(S_true[ 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std, m, n, h);gca().invert_yaxis()
    subplot(132)
    title("predicted pressure")
    visualize_scalar_on_fvm_points(S[ 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n]  .* p_std, m, n, h);gca().invert_yaxis()
    subplot(133)
    title("difference in pressure")
    visualize_scalar_on_fvm_points(S[ 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std .- S_true[ 2*(m+1)*(n+1)+1:2*(m+1)*(n+1)+m*n] .* p_std,  m, n, h);gca().invert_yaxis()
    tight_layout()
    savefig("xzchip_figures1/xzchipv0_nn_pres$k.png")

    
    figure(figsize=(15,4))
    subplot(131)
    title("exact temperature")
    visualize_scalar_on_fem_points(S_true[ 2*(m+1)*(n+1)+m*n+1:end] .* T_infty .+ T_infty, m, n, h);gca().invert_yaxis()
    subplot(132)
    title("predicted temperature")
    visualize_scalar_on_fem_points(S[ 2*(m+1)*(n+1)+m*n+1:end] .* T_infty .+ T_infty, m, n, h);gca().invert_yaxis()
    subplot(133)
    title("difference in temperature")
    visualize_scalar_on_fem_points(S[ 2*(m+1)*(n+1)+m*n+1:end]  .* T_infty .- S_true[2*(m+1)*(n+1)+m*n+1:end] .* T_infty, m, n, h);gca().invert_yaxis()
    tight_layout()
    savefig("xzchip_figures1/xzchipv0_nn_temp$k.png")

    figure(figsize=(15,4))
    subplot(131)
    xx = chip_left : h : chip_right
    yy = chip_top : h : chip_bottom
    k_chip_exact_2d = reshape(k_chip_exact, length(yy), length(xx))
    pcolor(xx, yy, k_chip_exact_2d)
    gca().set_aspect(1)
    colorbar()
    title("exact chip conductivity")
    

    subplot(132)
    k_chip_ = run(sess, k_chip)
    k_chip_2d = reshape(k_chip_, length(yy), length(xx))
    pcolor(xx, yy, k_chip_2d)
    gca().set_aspect(1)
    colorbar()
    title("predicted chip conductivity")

    subplot(133)
    pcolor(xx, yy, k_chip_2d .- k_chip_exact_2d)
    gca().set_aspect(1)
    colorbar()
    title("difference in chip conductivity")

    tight_layout()
    savefig("xzchip_figures1/xzchipv0_nn_cond$k.png")

end

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, zeros(m*n+3*(m+1)*(n+1)))

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, 2*(m+1)*(n+1)+m*n+(m+1)*(n+1)))

# sess = Session(); init(sess)
# output = run(sess, S)
V_computed = S[end, :]

V_data = matread("xzchipv0_fn_data.mat")["V"]

sample_size = 100
idx = rand(1:(m+1)*(n+1), sample_size)
idx = [idx; (m+1)*(n+1) .+ idx; 2*(m+1)*(n+1)+m*n .+ idx] # observe velocity and temperature
observed_data = V_data[idx]

noise = false
noise_level = 0.05
if noise
    noise_ratio = (1 - noise_level) .+ 2 * noise_level * rand(Float64, size(observed_data)) # uniform on (1-noise_level, 1+noise_level)
    observed_data = observed_data .* noise_ratio
end

loss = mean((V_computed[idx] .- observed_data)^2)
loss = loss * 1e10
# ---------------------------------------------------
# create a session and run 
max_iter = 1
sess = Session(); init(sess)

for k = 1:100
    loss_ = BFGS!(sess, loss, max_iter)
    matwrite("xzchip_figures1/loss$k.mat", Dict("L"=>loss_))
    close("all"); semilogy(loss_); title("loss vs. iteration")
    savefig("xzchip_figures1/loss$k.png")
    plot_velo_pres_temp_cond(k)
    ADCME.save(sess, "xzchip_figures1/nn$k.mat")
end

