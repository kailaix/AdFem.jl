using Revise
using PoreFlow
using LinearAlgebra
using PyPlot
using DelimitedFiles

mmesh = Mesh(8, 8 , 1/8)
DOF = zeros(Int64, 2mmesh.ndof)
for i = 1:mmesh.ndof
    DOF[i] = 2*i-1
    DOF[i+mmesh.ndof] = 2*i
end
u = readdlm("fenics/u.txt")[:]
u = u[DOF]
# u = ones(2mmesh.ndof)

μ = ones(get_ngauss(mmesh))
λ = ones(get_ngauss(mmesh))
ψ, J = neo_hookean(u, μ, λ, mmesh)


F = readdlm("fenics/F.txt")
S = readdlm("fenics/S.txt")
@info maximum(abs.(ψ - F[DOF]))
@info maximum(abs.(J - S[DOF, DOF]))

ψ, J = neo_hookean(constant(u), μ, λ, mmesh)
sess = Session(); init(sess)
@info maximum(abs.(run(sess, ψ) - F[DOF]))
@info maximum(abs.(run(sess, J) - S[DOF, DOF]))
