using Revise
using AdFem
using PyPlot 

# forward computation
mmesh = Meh(joinpath(PDATA, "twoholes_large.stl"))
xy = gauss_nodes(mmesh)
κ = @. sin(xy[:,1]) * (1+xy[:,2]^2) + 1.0
f = 1e5 * @. xy[:,1] + xy[:,2]
K = compute_fem_laplace_matrix1(κ, mmesh)
F = compute_fem_source_term1(f, mmesh)
bdnode = bcnode(mmesh)
K, F = impose_Dirichlet_boundary_conditions(K, F, bdnode, zeros(length(bdnode)))
sol = K\F

# inverse modeling 
nn_κ = squeeze(fc(xy, [20,20,20,1])) + 1
K = compute_fem_laplace_matrix1(nn_κ, mmesh)
F = compute_fem_source_term1(f, mmesh)
bdnode = bcnode(mmesh)
K, F = impose_Dirichlet_boundary_conditions(K, F, bdnode, zeros(length(bdnode)))
nn_sol = K\F
loss = sum((nn_sol - sol)^2)

sess = Session(); init(sess)
BFGS!(sess, loss)

nn_val = run(sess, nn_κ)
close("all")
figure(figsize=(18,5))
subplot(131)
visualize_scalar_on_gauss_points(κ, mmesh)
title("Reference")
subplot(132)
visualize_scalar_on_gauss_points(nn_val, mmesh)
title("DNN")
subplot(133)
visualize_scalar_on_gauss_points(abs.(κ-nn_val), mmesh)
title("Absolute Difference")
savefig("poisson_kappa.png")
close("all")
visualize_scalar_on_fem_points(sol, mmesh)
savefig("poisson_solution.png")