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

Ic, DIc, Jc, DJc = neo_hookean(u, mmesh)


F = readdlm("fenics/F.txt")
S = readdlm("fenics/S.txt")
@info maximum(abs.(Ic - F[DOF]))
@info maximum(abs.(DIc - S[DOF, DOF]))



F = readdlm("fenics/F1.txt")
S = readdlm("fenics/S1.txt")
@info maximum(abs.(Jc - F[DOF]))
@info maximum(abs.(DJc - S[DOF, DOF]))