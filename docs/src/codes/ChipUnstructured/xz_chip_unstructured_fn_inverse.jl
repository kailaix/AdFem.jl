using LinearAlgebra
using MAT
using PoreFlow
using PyPlot
matplotlib.use("agg") # or try "macosx"
using SparseArrays

trialnum = 1
include("chip_unstructured_geometry.jl")
include("plot_inverse_one_iter.jl")

k_mold = 0.014531
k_chip_ref = 2.60475
k_air = 0.64357

function k_exact(x, y)
    k_mold + 1000 * k_chip_ref * (x-0.49)^2 / (1 + x^2)
end

function k_nn(x, y)
    out = abs(fc(x, [20,20,20,1])) .+ k_mold
    squeeze(out)
end

nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

NT = 15    # number of iterations for Newton's method, 8 is good for m=400

xy = mesh.nodes 
xy2 = zeros(mesh.nedge, 2)
for i = 1:mesh.nedge
    xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
end
xy = [xy;xy2]

x, y = xy[chip_fem_idx, 1], xy[chip_fem_idx, 2]
k_chip = k_nn(x, y)

k_fem = k_air * constant(ones(ndof))
k_fem = scatter_update(k_fem, solid_fem_idx, k_mold * ones(length(solid_fem_idx)))
k_fem = scatter_update(k_fem, chip_fem_idx, k_chip)
kgauss = dof_to_gauss_points(k_fem, mesh)  # try change to include k on edges

heat_source_fem = zeros(ndof)
heat_source_fem[chip_fem_top_idx] .= power_source
# heat_source_fem[chip_fem_idx] .= power_source
heat_source_gauss = dof_to_gauss_points(heat_source_fem, mesh)  # try change to include k on edges

B = constant(compute_interaction_matrix(mesh)) # function not exist
# TODO: add unstructured mesh version of compute_interaction_matrix

# compute F
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(ngauss)), mesh))
heat_source = compute_fem_source_term1(constant(heat_source_gauss), mesh)

LaplaceK = constant(compute_fem_laplace_matrix1(kgauss, mesh))
# xy = fem_nodes(m, n, h)
# x, y = xy[:, 1], xy[:, 2]
# k = @. k_nn(x, y); k=stack(k)
# kgauss = dof_to_gauss_points(k, mesh)
# LaplaceK = compute_fem_laplace_matrix1(kgauss, mesh)

# apply Dirichlet to velocity and temperature; set left bottom two points to zero to fix rank deficient problem for pressure
bd = [bd; bd .+ ndof; 
      fvm_bd .+ 2*ndof; 
     bd .+ (2*ndof+nelem)]

# add solid region into boundary condition for u, v, p, i.e. exclude solid when solving Navier Stokes
bd = [bd; solid_fem_idx; solid_fem_idx .+ ndof; solid_fvm_idx .+ 2*ndof]

function compute_residual(S)
    u, v, p, T = S[1:ndof], 
        S[ndof+1:2*ndof], 
        S[2*ndof+1:2*ndof+nelem],
        S[2*ndof+nelem+1:end]
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)

    ugauss = dof_to_gauss_points(u, mesh)
    vgauss = dof_to_gauss_points(v, mesh)
    ux, uy, vx, vy = grad_u[:,1], grad_u[:,2], grad_v[:,1], grad_v[:,2]

    interaction = compute_interaction_term(p, mesh) # julia kernel needed
    f1 = compute_fem_source_term1(ugauss.*ux, mesh)
    f2 = compute_fem_source_term1(vgauss.*uy, mesh)
    f3 = -interaction[1:ndof]
    f4 = Laplace*u 
    # f5 = -F1
    F = f1 + f2 + f3 + f4 #+ f5 

    g1 = compute_fem_source_term1(ugauss.*vx, mesh)
    g2 = compute_fem_source_term1(vgauss.*vy, mesh)
    g3 = -interaction[ndof+1:end]    
    g4 = Laplace*v 
    # g5 = -F2
    T_gauss = dof_to_gauss_points(T, mesh)
    buoyance_term = - buoyance_coef * compute_fem_source_term1(T_gauss, mesh)

    G = g1 + g2 + g3 + g4 + buoyance_term #+ g5

    H0 = -B * [u;v] # + H

    T0 = LaplaceK * T + compute_fem_advection_matrix1(ugauss,vgauss, mesh) * T - heat_source
    R = [F;G;H0;T0]
    # return R  # original
    # return[F;G;H0;constant(zeros(ndof))]  # fix T=0
    # return[F;G;constant(zeros(nelem));constant(zeros(ndof))]  # fix p=0 and T=0
end

function compute_jacobian(S)
    u, v, p, T = S[1:ndof], 
        S[ndof+1:2*ndof], 
        S[2*ndof+1:2*ndof+nelem],
        S[2*ndof+nelem+1:end]
        
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)

    ugauss = dof_to_gauss_points(u, mesh)
    vgauss = dof_to_gauss_points(v, mesh)
    ux, uy, vx, vy = grad_u[:,1], grad_u[:,2], grad_v[:,1], grad_v[:,2]

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
    DU_TX = constant(compute_fem_mass_matrix1(Tx, mesh))       # (m+1)*(n+1), (m+1)*(n+1)
    DV_TY = constant(compute_fem_mass_matrix1(Ty, mesh))       # (m+1)*(n+1), (m+1)*(n+1)

    T_mat = constant(compute_fem_mass_matrix1(-buoyance_coef * constant(ones(ndof)), mesh))
    T_mat = [SparseTensor(spzeros(ndof, ndof)); T_mat]

    J0 = [Fu Fv
          Gu Gv]

    J1 = [J0 -B' T_mat
        -B spdiag(zeros(size(B,1))) SparseTensor(spzeros(nelem, ndof))]
    
    # N = 2 * ndof + nelem 
    J = [J1 
        [DU_TX DV_TY SparseTensor(spzeros(ndof, nelem)) M]]
        
    # return [J[1:2*ndof+nelem, 1:2*ndof+nelem]  SparseTensor(spzeros(2*ndof+nelem, ndof))
    # SparseTensor(spzeros(ndof,2*ndof+nelem)) spdiag(ones(ndof))] # fix T=0

    # return [J[1:2*ndof, 1:2*ndof]  SparseTensor(spzeros(2*ndof, nelem+ndof))
    # SparseTensor(spzeros(nelem+ndof,2*ndof)) spdiag(ones(nelem+ndof))]  # fix p=0 and T=0
end

function solve_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, mesh)
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
    S_new = solve_one_step(S)
    S_arr = write(S_arr, i, S_new)
    return i+1, S_arr
end

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, zeros(nelem+3*ndof))

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, nelem+3*ndof))

S_computed = S[end, :]
S_data = matread("fn$trialnum/xz_chip_unstructured_data.mat")["V"]

# sample_size = 20
# idx = rand(1:ndof, sample_size)
# idx = [idx; ndof .+ idx; 2*ndof+nelem .+ idx] # observe velocity and temperature
# observed_data = S_data[idx]

# noise = false
# noise_level = 0.05
# if noise
#     noise_ratio = (1 - noise_level) .+ 2 * noise_level * rand(Float64, size(observed_data)) # uniform on (1-noise_level, 1+noise_level)
#     observed_data = observed_data .* noise_ratio
# end

loss = mean((S_computed .- S_data)^2)
loss = loss * 1e10
# ---------------------------------------------------
# create a session and run 
max_iter = 1
sess = Session(); init(sess)

for k = 1:10000
    loss_ = BFGS!(sess, loss, max_iter)
    printstyled("[#iter $k] loss=$loss_\n", color=:green)
    matwrite("fn$trialnum/loss$k.mat", Dict("L"=>loss_))
    close("all"); semilogy(loss_); title("loss vs. iteration")
    savefig("fn$trialnum/nn_loss$k.png")
    plot_velo_pres_temp_cond(k)
    ADCME.save(sess, "fn$trialnum/nn_sess$k.mat")
end
