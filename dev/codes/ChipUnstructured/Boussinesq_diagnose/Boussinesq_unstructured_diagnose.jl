using ADCME
using PoreFlow
using LinearAlgebra
using MAT
using PyPlot; matplotlib.use("agg")
using SparseArrays

function k_exact(x,y)
    1 + x^2 + x / (1+y^2)
end

θ0 = [1.0; 1.0; 1.0]
θ = Variable([1.0; 1.0; 1.0])
paramA = θ[1]
paramB = θ[2]
paramC = θ[3]

# RBF2D

function k_func(x,y)
    1 + paramA * x^2 + paramB * x / (1+ paramC * y^2)
end

include("Boussinesq_common.jl")

m = 20
n = 20
h = 1/n

mesh = Mesh(m, n, h, degree=2)
nnode = mesh.nnode
nedge = mesh.nedge
ndof = mesh.ndof
nelem = mesh.nelem
ngauss = get_ngauss(mesh)

kgauss = eval_f_on_gauss_pts(k_func, mesh); kgauss=stack(kgauss)
LaplaceK = compute_fem_laplace_matrix1(kgauss, mesh)

S_computed = S[end, :]
S_data = matread("data.mat")["V"]

loss =  mean((S_computed .- S_data)^2)
loss = loss * 1e10

sess = Session(); init(sess)

# @info run(sess, loss, θ=>θ0)
# lineview(sess, θ, loss, θ0, zeros(3))
# savefig("lineview.png")
# gradview(sess, θ, loss, zeros(3))
# savefig("gradview.png")

# gradview(sess, θ, sum(kgauss), zeros(3))
# savefig("gradview_kgauss.png")

# gradview(sess, θ, sum(values(LaplaceK)^2), zeros(3))
# savefig("gradview_LaplaceK.png")
# @info S[2, :]

# gradview(sess, θ, sum(S[2, :]^2), zeros(3))
# savefig("gradview_S2.png")

gradview(sess, θ, sum(S[3, :]^2), zeros(3))
savefig("gradview_S3.png")

# gradview(sess, θ, sum(S_computed^2), zeros(3))
# savefig("gradview_S_computed.png")
