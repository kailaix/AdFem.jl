using ADCME
using AdFem

include("../chip_unstructured_solver.jl")
include("../chip_unstructured_geometry.jl")

k_mold = 0.014531
k_chip_ref = 2.60475
k_air = 0.64357

function k_exact(x, y)
    k_mold + k_chip_ref * exp( -(x-0.5)^2 / 0.001)
end

k_chip = eval_f_on_dof_pts(k_exact, mesh)[chip_fem_idx]

nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

NT = 15    # number of iterations for Newton's method

heat_source_fem = zeros(ndof)
heat_source_fem[chip_fem_top_idx] .= power_source
heat_source_gauss = dof_to_gauss_points(heat_source_fem, mesh)

B = constant(compute_interaction_matrix(mesh))
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(ngauss)), mesh))
heat_source = compute_fem_source_term1(constant(heat_source_gauss), mesh)

# apply Dirichlet to velocity and temperature; set left bottom two points to zero to fix rank deficient problem for pressure
bd = [bd; bd .+ ndof; 
      fvm_bd .+ 2*ndof; 
      bd .+ (2*ndof+nelem)]

# add solid region into boundary condition for u, v, p, i.e. exclude solid when solving Navier Stokes
bd = [bd; solid_fem_idx; solid_fem_idx .+ ndof; solid_fvm_idx .+ 2*ndof]

S0 = constant(zeros(nelem+3*ndof))
S = solve_navier_stokes(S0, NT, k_chip)

sess = Session(); init(sess)
output = run(sess, S)

matwrite("data3.mat", 
    Dict(
        "V"=>output[end, :]
    ))


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
