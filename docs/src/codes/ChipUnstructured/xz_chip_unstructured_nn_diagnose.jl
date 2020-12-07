using ADCME
using AdFem

include("chip_unstructured_solver.jl")
include("chip_unstructured_geometry.jl")
include("plot_inverse_one_iter.jl")

trialnum = 3

k_mold = 0.014531
k_chip_ref = 2.60475
k_air = 0.64357

# θ0 = [2.60475; 0.49; 1.0]
# θ = Variable([2.60475; 0.49; 1.0])
# paramA = θ[1]
# paramB = θ[2]
# paramC = θ[3]

function k_exact(x, y)
    k_mold + 1000 * k_chip_ref * (x-0.49)^2 / (1 + x^2)
end

function k_nn(x, y, θ)
    constant(@. k_exact(x, y)) +  fc(x, [20,20,20,1], θ)
end

nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

NT = 14    # number of iterations for Newton's method, 8 is good for m=400

n = 901  #ae_num([20,20,20,1]) => 861?
θ = Variable(randn(n))
θ = θ ./ norm(θ)

xy = mesh.nodes 
xy2 = zeros(mesh.nedge, 2)
for i = 1:mesh.nedge
    xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
end
xy = [xy;xy2]

x, y = xy[chip_fem_idx, 1], xy[chip_fem_idx, 2]
k_chip =  k_nn(x, y, θ)
k_chip_exact = constant( @. k_exact(x, y) )

heat_source_fem = zeros(ndof)
heat_source_fem[chip_fem_top_idx] .= power_source
heat_source_gauss = dof_to_gauss_points(heat_source_fem, mesh)
heat_source = compute_fem_source_term1(constant(heat_source_gauss), mesh)

B = constant(compute_interaction_matrix(mesh))
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(ngauss)), mesh))

# apply Dirichlet to velocity and temperature; set left bottom two points to zero to fix rank deficient problem for pressure
bd = [bd; bd .+ ndof; 
      fvm_bd .+ 2*ndof; 
      bd .+ (2*ndof+nelem)]

# add solid region into boundary condition for u, v, p, i.e. exclude solid when solving Navier Stokes
bd = [bd; solid_fem_idx; solid_fem_idx .+ ndof; solid_fvm_idx .+ 2*ndof]

S0 = constant(zeros(nelem+3*ndof))
S = solve_navier_stokes(S0, NT, k_chip)
S_computed = S[end, :]
S_data = matread("fn1/xz_chip_unstructured_data.mat")["V"]

loss =  mean((S_computed .- S_data)^2)
loss = loss * 1e10
sess = Session(); init(sess)

# ---------------------------------------------------
n = 901
θ0 = zeros(n)
@info run(sess, loss, θ=>θ0)
lineview(sess, θ, loss, θ0, ones(n))
savefig("fn$trialnum/lineview.png")
gradview(sess, θ, loss, zeros(n))
savefig("fn$trialnum/gradview.png")
