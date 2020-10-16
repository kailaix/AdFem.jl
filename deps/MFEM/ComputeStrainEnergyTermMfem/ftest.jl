using Revise
using AdFem 
using PyPlot 
using LinearAlgebra

mmesh = Mesh(10, 10, 0.1)
u = rand(2mmesh.ndof)
ε = eval_strain_on_gauss_pts(u, mmesh)
K = [1.0 0.0 0.0;0.0 1.0 0.0; 0.0 0.0 0.5]
σ = zeros(size(ε)...)
for i = 1:get_ngauss(mmesh)
    σ[i,:] = K * ε[i,:]
end
e1 = compute_strain_energy_term(σ, mmesh)

Q = compute_fem_stiffness_matrix(K, mmesh)
e2 = Q * u 

@info norm(e1 - e2 )


mmesh = Mesh(10, 10, 0.1, degree=2)
u = rand(2mmesh.ndof)
ε = eval_strain_on_gauss_pts(u, mmesh)
K = [1.0 0.0 0.0;0.0 1.0 0.0; 0.0 0.0 0.5]
σ = zeros(size(ε)...)
for i = 1:get_ngauss(mmesh)
    σ[i,:] = K * ε[i,:]
end
e1 = compute_strain_energy_term(σ, mmesh)

Q = compute_fem_stiffness_matrix(K, mmesh)
e2 = Q * u 

@info norm(e1 - e2 )
