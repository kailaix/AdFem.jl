using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics
using MAT 

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))
mmesh = Mesh(mmesh.nodes * 10, mmesh.elems, -1, 2)


ν = sigmoid(Variable(zeros(mmesh.nelem)) ) * 0.002
ν = repeat(ν, 1, get_ngauss(mmesh)÷mmesh.nelem)
ν = reshape(ν, (-1,))


K = constant(compute_fem_laplace_matrix(ν, mmesh))
B = constant(compute_interaction_matrix(mmesh))
Z = [K -B'
    -B spzero(size(B,1))]

bd1 = bcnode((x,y)->y>0.23-1e-5, mmesh)
bd2 = bcnode((x,y)->y<=0.23-1e-5, mmesh)
bd = [bd1;bd2]
bd = [bd; bd .+ mmesh.ndof; 2mmesh.ndof + 1]


rhs = zeros(2mmesh.ndof + mmesh.nelem)
bdval = zeros(length(bd))
bdval[1:length(bd1)] .= 1.0
Z, rhs = impose_Dirichlet_boundary_conditions(Z, rhs, bd, bdval)
sol = Z\rhs 

U = matread("fenics/fwd2.mat")["u"]
loss = sum((sol[2mmesh.ndof+1:end] - U[2mmesh.ndof+1:end])^2)

sess = Session(); init(sess)

_loss = Float64[]
cb = (vs, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
    if mod(iter, 10)==1
        make_directory("pixel2")
        close("all")
        visualize_scalar_on_gauss_points(vs[1], mmesh)
        matwrite("pixel2/bwd3-$iter.mat", Dict("iter"=>iter,"loss"=>_loss, "nu"=>vs[1]))
        savefig("pixel2/bwd3_pixel$iter.png")
    end
end


BFGS!(sess, loss, vars = [ν], callback = cb)

