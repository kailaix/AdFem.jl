using LinearAlgebra
using MAT
using PoreFlow
using PyPlot
matplotlib.use("agg") # or try "macosx"
using SparseArrays

# geometry setup in domain [0,1]^2
solid_left = 0.48
solid_right = 0.52
solid_top = 0.5
solid_bottom = 0.52

chip_left = 0.48
chip_right = 0.52
chip_top = 0.5
chip_bottom = 0.505

delta=1e-6
delta2=1e-6

k_mold = 0.014531
k_chip = 2.60475
k_air = 0.64357
nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

filename = "mesh/CHT_2D.stl"
file_format = "stl"
mesh = Mesh(filename, file_format = file_format, degree=2)
mesh = Mesh(mesh.nodes ./ 0.0305, mesh.elems, -1, 2)
h = 0.1

# m = 100
# n = 100
# h = 1/n
# mesh = Mesh(m, n, h, degree=2)

nnode = mesh.nnode
nedge = mesh.nedge
ndof = mesh.ndof
nelem = mesh.nelem
ngauss = get_ngauss(mesh)

NT = 6    # number of iterations for Newton's method, 8 is good for m=400


# compute solid indices and chip indices
solid_fem_idx = Array{Int64, 1}([])
solid_fvm_idx = Array{Int64, 1}([])
chip_fem_idx = Array{Int64, 1}([])
# chip_fvm_idx = Array{Int64, 1}([])
chip_fem_top_idx = Array{Int64, 1}([])
bd = Array{Int64, 1}([])
fvm_bd = Array{Int64, 1}([])



for j = 1:nnode
    nodex, nodey = mesh.nodes[j, 1], mesh.nodes[j, 2]
    if nodex >= solid_left-delta2 && nodex <= solid_right+delta2 && nodey >= solid_top-delta2 && nodey <= solid_bottom+delta2
        # print(i, j)
        global solid_fem_idx = [solid_fem_idx; j]
        if nodex >= chip_left-delta2 && nodex <= chip_right+delta2 && nodey >= chip_top-delta2 && nodey <= chip_bottom+delta2
            global chip_fem_idx = [chip_fem_idx; j]
            if nodey <= chip_top+delta2
                global chip_fem_top_idx = [chip_fem_top_idx; j]
            end
        end
    end
    if abs(nodex-0.0) <= delta || abs(nodex-1.0) <= delta || abs(nodey-0.0) <= delta || abs(nodey-1.0) <= delta
        global bd = [bd; j]
    end
end

# fix chip_fem_top_idx
if size(chip_fem_top_idx, 1) == 0
    chip_fem_top_idx = chip_fem_idx
end

for j = 1:nedge
    edgex, edgey = (mesh.nodes[mesh.edges[j, 1], :] .+ mesh.nodes[mesh.edges[j, 2], :]) ./ 2
    if edgex >= solid_left-delta2 && edgex <= solid_right+delta2 && edgey >= solid_top-delta2 && edgey <= solid_bottom+delta2
        # print(i, j)
        global solid_fem_idx = [solid_fem_idx; nnode + j]
        if edgex >= chip_left-delta2 && edgex <= chip_right+delta2 && edgey >= chip_top-delta2 && edgey <= chip_bottom+delta2
            global chip_fem_idx = [chip_fem_idx; nnode + j]
            if edgex >= chip_left-delta2 && edgex <= chip_right+delta2 && edgey >= chip_top-delta2 && edgey <= chip_top+delta2
                global chip_fem_top_idx = [chip_fem_top_idx; nnode + j]
            end
        end
    end
    if abs(edgex-0.0) <= delta || abs(edgex-1.0) <= delta || abs(edgey-0.0) <= delta || abs(edgey-1.0) <= delta
        global bd = [bd; nnode + j]
    end
end

# bd_vec = zeros(ndof)
# for i in 1:length(bd)
#     bd_vec[ bd[i] ] = 1.0
# end
# figure();visualize_scalar_on_fem_points(bd_vec[1:nnode],mesh,with_mesh=true);savefig("bd.pdf")

# solid_vec = zeros(ndof)
# for i in 1:length(solid_fem_idx)
#     solid_vec[ solid_fem_idx[i] ] = 1.0
# end
# for i in 1:length(chip_fem_idx)
#     solid_vec[ chip_fem_idx[i] ] = 2.0
# end
# figure();visualize_scalar_on_fem_points(solid_vec[1:nnode],mesh,with_mesh=true);savefig("solid.pdf")

gaussxy = gauss_nodes(mesh)

for i in 1:nelem
    gaussx, gaussy = mean(gaussxy[6*i-5: 6*i, 1]), mean(gaussxy[6*i-5: 6*i, 2])
    # print("\nx=",gaussx, " y=",gaussy)
    if gaussx >= solid_left-delta2 && gaussx <= solid_right+delta2 && gaussy >= solid_top-delta2 && gaussy <= solid_bottom+delta2
        global solid_fvm_idx = [solid_fvm_idx; i]
    end
    if abs(gaussy - 0.0) <= h/2 + delta && ( abs(gaussx - 0.0) <= h + delta || abs(gaussx - 1.0) <= h + delta )
        global fvm_bd = [fvm_bd; i]
    end
end

k_fem = k_air * constant(ones(ndof))
k_fem = scatter_update(k_fem, solid_fem_idx, k_mold * ones(length(solid_fem_idx)))
k_fem = scatter_update(k_fem, chip_fem_idx, k_chip * ones(length(chip_fem_idx)))
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

sess = Session(); init(sess)
output = run(sess, S)
# J = run(sess, J); rank(J)
# J1 = run(sess, J1); rank(J1)
# J0 = run(sess, J0); rank(J0)

# matwrite("xz_chip_unstructured_data.mat", 
#     Dict(
#         "V"=>output[end, :]
#     ))


# TODO: fix plot
u_out, v_out, p_out, T_out = output[NT+1,1:nnode], output[NT+1,ndof+1:ndof+nnode], 
                             output[NT+1,2*ndof+1:2*ndof+nelem],output[NT+1,2*ndof+nelem+1:2*ndof+nelem+nnode]

figure(figsize=(10,10))
subplot(221)
title("u velocity")
visualize_scalar_on_fem_points(u_out .* u_std, mesh);#gca().invert_yaxis()
subplot(222)
title("v velocity")
visualize_scalar_on_fem_points(v_out .* u_std, mesh);#gca().invert_yaxis()
subplot(223)
visualize_scalar_on_fvm_points(p_out .* p_std, mesh);#gca().invert_yaxis()
title("pressure")
subplot(224)
title("temperature")
visualize_scalar_on_fem_points(T_out.* T_infty .+ T_infty, mesh);#gca().invert_yaxis()
tight_layout()
savefig("forward_solution_unstructured.pdf")

print("Solution range:",
    "\n [u velocity] \t min:", minimum(u_out .* u_std), ",\t max:", maximum(u_out .* u_std),
    "\n [v velocity] \t min:", minimum(v_out .* u_std), ",\t max:", maximum(v_out .* u_std),
    "\n [pressure]   \t min:", minimum(p_out .* p_std), ",\t max:", maximum(p_out .* p_std),
    "\n [temperature]\t min:", minimum(T_out.* T_infty .+ T_infty), ",\t\t\t max:", maximum(T_out.* T_infty .+ T_infty))
