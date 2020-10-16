nu = 0.01
buoyance_coef = 1.0

function u_exact(x,y)
    x*(1-x)*y*(1-y)
end

function v_exact(x,y)
    x*(1-x)*y*(1-y)
end

function p_exact(x,y)
    x*(1-x)*y*(1-y)
end

function t_exact(x,y)
    x*(1-x)*y*(1-y)
end

function ffunc_(x, y)
    x*y*(1 - x)*(1 - y)*(-x*y*(1 - x) + x*(1 - x)*(1 - y)) + x*y*(1 - x)*(1 - y)*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - x*y*(1 - y) + 0.02*x*(1 - x) + y*(1 - x)*(1 - y) + 0.02*y*(1 - y)    
end

function gfunc_(x, y)
    x*y*(1 - x)*(1 - y)*(-x*y*(1 - x) + x*(1 - x)*(1 - y)) + x*y*(1 - x)*(1 - y)*(-x*y*(1 - y) + y*(1 - x)*(1 - y)) - buoyance_coef * x*y*(1 - x)*(1 - y) - x*y*(1 - x) + x*(1 - x)*(1 - y) + 0.02*x*(1 - x) + 0.02*y*(1 - y)
end

function hfunc_(x,y)
    -x*y*(1 - x) - x*y*(1 - y) + x*(1 - x)*(1 - y) + y*(1 - x)*(1 - y)
end

function heat_source_func(x, y)
    (2*x^2*y*(x - 1)*(2*y - 1) + x*y*(x - 1)*(y - 1)*(y^2 + 1)^2*(x*(x - 1)*(2*y - 1) + y*(2*x - 1)*(y - 1)) - (y^2 + 1)*(2*x*(x - 1)*(x + (x^2 + 1)*(y^2 + 1)) + 2*y*(x + (x^2 + 1)*(y^2 + 1))*(y - 1) + y*(2*x - 1)*(y - 1)*(2*x*(y^2 + 1) + 1)))/(y^2 + 1)^2
end

m = 20
n = 20
h = 1/n

mesh = Mesh(m, n, h, degree=2)
nnode = mesh.nnode
nedge = mesh.nedge
ndof = mesh.ndof
nelem = mesh.nelem
ngauss = get_ngauss(mesh)

F1 = compute_fem_source_term1(eval_f_on_gauss_pts(ffunc_, mesh), mesh)
F2 = compute_fem_source_term1(eval_f_on_gauss_pts(gfunc_, mesh), mesh)
H = get_area(mesh) .* eval_f_on_fvm_pts(hfunc_, mesh)
B = constant(compute_interaction_matrix(mesh))

# compute F
Laplace = compute_fem_laplace_matrix1(nu * constant(ones(ngauss)), mesh)
heat_source = eval_f_on_gauss_pts(heat_source_func, mesh)
heat_source = constant(compute_fem_source_term1(heat_source, mesh))

kgauss = eval_f_on_gauss_pts(k_func, mesh); kgauss=stack(kgauss)
LaplaceK = constant(compute_fem_laplace_matrix1(kgauss, mesh))

# xy = fem_nodes(mesh)
# x, y = xy[:, 1], xy[:, 2]
# k = @. k_nn(x, y); k=stack(k)
# kgauss = dof_to_gauss_points(k, mesh)
# LaplaceK = compute_fem_laplace_matrix1(kgauss, mesh)

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
    f5 = -F1
    F = f1 + f2 + f3 + f4 + f5 

    g1 = compute_fem_source_term1(ugauss.*vx, mesh)
    g2 = compute_fem_source_term1(vgauss.*vy, mesh)
    g3 = -interaction[ndof+1:end]    
    g4 = Laplace*v 
    g5 = -F2

    T_gauss = dof_to_gauss_points(T, mesh)
    buoyance_term = - buoyance_coef * compute_fem_source_term1(T_gauss, mesh)

    G = g1 + g2 + g3 + g4 + g5 + buoyance_term

    H0 = -B * [u;v] + H

    T0 = LaplaceK * T + compute_fem_advection_matrix1(ugauss,vgauss, mesh) * T - heat_source
    R = [F;G;H0;T0]
    return R
    # return [F; G; constant(zeros(nelem)); constant(zeros(ndof)) ]
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

    T_mat = constant(compute_fem_mass_matrix1(-buoyance_coef * constant(ones(ngauss)), mesh))
    T_mat = [SparseTensor(spzeros(ndof, ndof)); T_mat]

    J0 = [Fu Fv
          Gu Gv]

    J1 = [J0 -B' T_mat
        -B spdiag(zeros(size(B,1))) SparseTensor(spzeros(nelem, ndof))]
    
    J = [J1 
        [DU_TX DV_TY SparseTensor(spzeros(ndof, nelem)) M]]

    # return [J[1:2ndof, 1:2ndof] SparseTensor(spzeros(2ndof, nelem+ndof))
        # SparseTensor(spzeros(nelem+ndof, 2ndof))  spdiag(ones(nelem+ndof)) ]    
    end

NT = 8    # number of iterations for Newton's method

bd = Array{Int64, 1}([])
eps = 1e-6
for j = 1:nnode
    nodex, nodey = mesh.nodes[j, 1], mesh.nodes[j, 2]
    if abs(nodex-0.0) <= eps || abs(nodex-1.0) <= eps || abs(nodey-0.0) <= eps || abs(nodey-1.0) <= eps
        global bd = [bd; j]
    end
end
for j = 1:nedge
    edgex, edgey = (mesh.nodes[mesh.edges[j, 1], :] .+ mesh.nodes[mesh.edges[j, 2], :]) ./ 2
    if abs(edgex-0.0) <= eps || abs(edgex-1.0) <= eps || abs(edgey-0.0) <= eps || abs(edgey-1.0) <= eps
        global bd = [bd; nnode + j]
    end
end


fvm_bd = Array{Int64, 1}([])
elemxy = fvm_nodes(mesh)
for j = 1:nelem
    elemx, elemy = elemxy[j, 1], elemxy[j, 2]
    if abs(elemx-0.0) <= 2*h && abs(elemy-0.0) <= h
        global fvm_bd = [fvm_bd; j]
    end
    if abs(elemx-1.0) <= 2*h && abs(elemy-0.0) <= h
        global fvm_bd = [fvm_bd; j]
    end
end

bd = [bd; bd .+ ndof; fvm_bd .+ 2*ndof; bd .+ (2*ndof+nelem)] 
# only apply Dirichlet to velocity; set left bottom two points to zero to fix rank deficient problem for pressure


function solve_steady_cavityflow_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, residual = impose_Dirichlet_boundary_conditions(J, residual, bd, zeros(length(bd)))

    d = J\residual
    residual_norm = norm(residual)
    op = tf.print("residual norm", residual_norm)
    d = bind(d, op)
    S_new = S - d
    return S_new
end

function condition(i, S_arr)
    i <= NT
end

function body(i, S_arr)
    S = read(S_arr, i)
    op = tf.print("i=",i)
    i = bind(i, op)
    S_new = solve_steady_cavityflow_one_step(S)
    S_arr = write(S_arr, i+1, S_new)
    return i+1, S_arr
end

xy = fem_nodes(mesh)
x, y = xy[:,1], xy[:,2]
u0 = @. u_exact(x,y)
v0 = @. v_exact(x,y)
t0 = @. t_exact(x,y)


xy = fvm_nodes(mesh)
x, y = xy[:,1], xy[:,2]
p0 = @. p_exact(x,y)

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, zeros(nelem+3*ndof))

i = constant(1, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, nelem+3*ndof))