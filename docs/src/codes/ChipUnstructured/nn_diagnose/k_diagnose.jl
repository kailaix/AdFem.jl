using ADCME
using AdFem

include("../chip_unstructured_solver.jl")
include("../chip_unstructured_geometry.jl")

k_mold = 0.014531
k_chip_ref = 2.60475
k_air = 0.64357

function k_exact(x, y)
    k_mold + 1000 * k_chip_ref * (x-0.49)^2 / (1 + x^2)
end

function k_nn(x, y)
    out =  fc(x, [20,20,20,1])^2  .+ k_mold
    squeeze(out)
end

xy = mesh.nodes 
xy2 = zeros(mesh.nedge, 2)
for i = 1:mesh.nedge
    xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
end
xy = [xy;xy2]

x, y = xy[chip_fem_idx, 1], xy[chip_fem_idx, 2]
k_chip = k_nn(x, y)
k_chip_exact = @. k_exact(x, y)

loss =  mean((k_chip .- k_chip_exact)^2)
loss = loss * 1e10

# ---------------------------------------------------
_loss = Float64[]
cb = (vs, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
end

sess = Session(); init(sess)
BFGS!(sess, loss, vars = [k_chip], callback = cb)
