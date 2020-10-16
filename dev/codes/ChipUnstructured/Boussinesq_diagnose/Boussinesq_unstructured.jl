using ADCME
using AdFem
using LinearAlgebra
using MAT
using PyPlot; matplotlib.use("agg")
using SparseArrays

function k_func(x,y)
    1 + x^2 + x / (1+y^2)
end

include("Boussinesq_common.jl")

sess = Session(); init(sess)
output = run(sess, S)

matwrite("data.mat", 
    Dict(
        "V"=>output[end, :]
    ))

u_out, v_out, p_out, T_out = output[NT+1,1:nnode], output[NT+1,ndof+1:ndof+nnode], 
                             output[NT+1,2*ndof+1:2*ndof+nelem],output[NT+1,2*ndof+nelem+1:2*ndof+nelem+nnode]

figure(figsize=(25,10))
subplot(241)
title("u velocity")
visualize_scalar_on_fem_points(u_out, mesh)
subplot(245)
visualize_scalar_on_fem_points(u0, mesh)

subplot(242)
title("v velocity")
visualize_scalar_on_fem_points(v_out, mesh)
subplot(246)
visualize_scalar_on_fem_points(v0, mesh)

subplot(243)
visualize_scalar_on_fvm_points(p_out, mesh)
title("pressure")
subplot(247)
visualize_scalar_on_fvm_points(p0, mesh)
title("")

subplot(244)
title("temperature")
visualize_scalar_on_fem_points(T_out, mesh)
subplot(248)
visualize_scalar_on_fem_points(t0, mesh)

tight_layout()
savefig("data.png")
close("all")
