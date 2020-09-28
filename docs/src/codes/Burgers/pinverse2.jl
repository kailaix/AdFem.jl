include("common.jl")


function f(x, y)
    0.0001*(10/(1+x^2) + x * y + 10*y^2)
end



Δt = 0.01
mmesh = Mesh(30, 30, 1/30)
bdnode = bcnode(mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
# nu = constant(eval_f_on_gauss_pts(f, mmesh))


nu = abs(squeeze(fc(gauss_nodes(mmesh), [20,20,20,1]))) * 0.0001 

nodes = fem_nodes(mmesh)
u = @. sin(2π * nodes[:,1])
v = @. sin(2π * nodes[:,2])
u0 = [u;v]
u0[bdnode] .= 0.0
us = solve_burgers(u0, 10, nu)


U = matread("fenics/fwd2.mat")["U"]
loss = sum((us - U)^2)

_loss = Float64[]
cb = (vs, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
    if mod(iter, 10)==1
        close("all")
        visualize_scalar_on_gauss_points(vs[1], mmesh)
        matwrite("fenics/bwd2-$iter.mat", Dict("iter"=>iter,"loss"=>_loss, "nu"=>vs[1]))
        savefig("fenics/bwd2_nn$iter.png")
    end
end

# @info run(sess, loss)
sess = Session(); init(sess)
BFGS!(sess, loss, vars = [nu], callback = cb)
