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
nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

m = 50
n = 50
h = 1/n
mesh = Mesh(m, n, h)
NT = 1    # number of iterations for Newton's method, 8 is good for m=400


# compute solid indices and chip indices
solid_fem_idx = Array{Int64, 1}([])
solid_fvm_idx = Array{Int64, 1}([])
chip_fem_idx = Array{Int64, 1}([])
chip_fvm_idx = Array{Int64, 1}([])
chip_fem_top_idx = Array{Int64, 1}([])

nnode = size(mesh.nodes, 1)
nelem = size(mesh.elems, 1)

for j = 1:nnode
    nodex, nodey = mesh.nodes[j, 1], mesh.nodes[j, 2]
    if nodex >= solid_left-1e-9 && nodex <= solid_right+1e-9 && nodey >= solid_top-1e-9 && nodey <= solid_bottom+1e-9
        # print(i, j)
        global solid_fem_idx = [solid_fem_idx; j]
        if nodex >= chip_left-1e-9 && nodex <= chip_right+1e-9 && nodey >= chip_top-1e-9 && nodey <= chip_bottom+1e-9
            global chip_fem_idx = [chip_fem_idx; j]
            if nodex >= chip_left-1e-9 && nodex <= chip_right+1e-9 && nodey >= chip_top-1e-9 && nodey <= chip_top+1e-9
                global chip_fem_top_idx = [chip_fem_top_idx; j]
            end
        end
    end
end

gaussxy = gauss_nodes(mesh)

for i in 1:nelem
    gaussx, gaussy = (gaussxy[3*i-2, 1]+gaussxy[3*i-1, 1]+gaussxy[3*i, 1])/3.0, 
                      (gaussxy[3*i-2, 2]+gaussxy[3*i-1, 2]+gaussxy[3*i, 2])/3.0
    if gaussx >= solid_left-1e-9 && gaussx <= solid_right+1e-9 && gaussy >= solid_top-1e-9 && gaussy <= solid_bottom+1e-9
        # print(i, j)
        global solid_fvm_idx = [solid_fvm_idx; i]
    end
end

##########################################################################################################
# TODO: construct solid_fvm_index and chip_fvm_index

##########################################################################################################

k_fem = k_air * constant(ones(nnode))
k_fem = scatter_update(k_fem, solid_fem_idx, k_mold * ones(length(solid_fem_idx)))
k_fem = scatter_update(k_fem, chip_fem_idx, k_chip * ones(length(chip_fem_idx)))
kgauss = fem_to_gauss_points(k_fem, mesh)

heat_source_fem = zeros(nnode)
heat_source_fem[chip_fem_idx] .= power_source #/ h^2
heat_source_fem[chip_fem_top_idx] .= 82.46295
heat_source_gauss = fem_to_gauss_points(heat_source_fem, mesh)

B = constant(compute_interaction_matrix(mesh)) # function not exist
# TODO: add unstructured mesh version of compute_interaction_matrix

# compute F
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(nnode)), mesh))
heat_source = compute_fem_source_term1(constant(heat_source_gauss), mesh)

LaplaceK = constant(compute_fem_laplace_matrix1(kgauss, mesh))
# xy = fem_nodes(m, n, h)
# x, y = xy[:, 1], xy[:, 2]
# k = @. k_nn(x, y); k=stack(k)
# kgauss = fem_to_gauss_points(k, mesh)
# LaplaceK = compute_fem_laplace_matrix1(kgauss, mesh)


# TODO: fix bd
bd = Int64[]
for j = 1:m+1
    push!(bd, j)
    push!(bd, n*(m+1)+j)
end
for i = 2:n
    push!(bd, (i-1)*(m+1)+1)
    push!(bd, (i-1)*(m+1)+m+1)
end

# only apply Dirichlet to velocity; set left bottom two points to zero to fix rank deficient problem for pressure

bd = [bd; bd .+ nnode; 
     2*nnode+1: 2*nnode+2; 
     bd .+ (2*nnode+nelem)]

# add solid region into boundary condition for u, v, p
bd = [bd; solid_fem_idx; solid_fem_idx .+ nnode; solid_fvm_idx .+ 2*nnode]


function compute_residual(S)
    u, v, p, T = S[1:nnode], 
        S[nnode+1:2*nnode], 
        S[2*nnode+1:2*nnode+nelem],
        S[2*nnode+nelem+1:end]
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)

    ugauss = fem_to_gauss_points(u, mesh)
    vgauss = fem_to_gauss_points(v, mesh)
    ux, uy, vx, vy = grad_u[:,1], grad_u[:,2], grad_v[:,1], grad_v[:,2]

    interaction = compute_interaction_term(p, mesh) # julia kernel needed
    f1 = compute_fem_source_term1(ugauss.*ux, mesh)
    f2 = compute_fem_source_term1(vgauss.*uy, mesh)
    f3 = -interaction[1:nnode]
    f4 = Laplace*u 
    # f5 = -F1
    F = f1 + f2 + f3 + f4 #+ f5 

    g1 = compute_fem_source_term1(ugauss.*vx, mesh)
    g2 = compute_fem_source_term1(vgauss.*vy, mesh)
    g3 = -interaction[nnode+1:end]    
    g4 = Laplace*v 
    # g5 = -F2
    T_gauss = fem_to_gauss_points(T, mesh)
    buoyance_term = - buoyance_coef * compute_fem_source_term1(T_gauss, mesh)

    G = g1 + g2 + g3 + g4 + buoyance_term #+ g5

    H0 = -B * [u;v] # + H

    T0 = LaplaceK * T + compute_fem_advection_matrix1(ugauss,vgauss, mesh) * T - heat_source
    R = [F;G;H0;T0]
    return R
end

function compute_jacobian(S)
    u, v, p, T = S[1:nnode], 
        S[nnode+1:2*nnode], 
        S[2*nnode+1:2*nnode+nelem],
        S[2*nnode+nelem+1:end]
        
    grad_u = eval_grad_on_gauss_pts1(u, mesh)
    grad_v = eval_grad_on_gauss_pts1(v, mesh)

    ugauss = fem_to_gauss_points(u, mesh)
    vgauss = fem_to_gauss_points(v, mesh)
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

    T_mat = constant(compute_fem_mass_matrix1(-buoyance_coef * constant(ones(nnode)), mesh))
    T_mat = [SparseTensor(spzeros(nnode, nnode)); T_mat]

    J0 = [Fu Fv
          Gu Gv]

    J1 = [J0 -B' T_mat
        -B spdiag(zeros(size(B,1))) SparseTensor(spzeros(nelem, nnode))]
    
    N = 2 * nnode + nelem 
    J = [J1 
        [DU_TX DV_TY SparseTensor(spzeros(nnode, nelem)) M]]
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
S_arr = write(S_arr, 1, zeros(nelem+3*nnode))

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, nelem+3*nnode))

sess = Session(); init(sess)
output = run(sess, S)
J = run(sess, J)
rank(J)

matwrite("xz_chip_unstructured_data.mat", 
    Dict(
        "V"=>output[end, :]
    ))


u0, v0, p0, t0 = zeros(nnode), zeros(nnode), zeros(nelem), zeros(nnode)



# TODO: fix plot
u_out, v_out, p_out, T_out = output[NT+1,:nnode], output[NT+1,nnode+1:2*nnode], 
                             output[NT+1,2*nnode+1:2*nnode+nelem],output[NT+1,2*nnode+nelem+1:end]

figure(figsize=(10,10))
subplot(221)
title("u velocity")
visualize_scalar_on_fem_points(u_out .* u_std, m, n, h);gca().invert_yaxis()
subplot(222)
title("v velocity")
visualize_scalar_on_fem_points(v_out .* u_std, m, n, h);gca().invert_yaxis()
subplot(223)
visualize_scalar_on_fvm_points(p_out .* p_std, m, n, h);gca().invert_yaxis()
title("pressure")
subplot(224)
title("temperature")
visualize_scalar_on_fem_points(T_out.* T_infty .+ T_infty, m, n, h);gca().invert_yaxis()
tight_layout()
savefig("forward_solution_unstructured.png")

print("Solution range:",
    "\n [u velocity] \t min:", minimum(u_out .* u_std), ",\t max:", maximum(u_out .* u_std),
    "\n [v velocity] \t min:", minimum(v_out .* u_std), ",\t max:", maximum(v_out .* u_std),
    "\n [pressure]   \t min:", minimum(p_out .* p_std), ",\t max:", maximum(p_out .* p_std),
    "\n [temperature]\t min:", minimum(T_out.* T_infty .+ T_infty), ",\t\t\t max:", maximum(T_out.* T_infty .+ T_infty))
