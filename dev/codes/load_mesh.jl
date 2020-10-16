using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics

if pdata == 1
    mmesh = Mesh(50, 50, 1/50)
elseif pdata == 2
    mmesh = Mesh(joinpath(PDATA, "pipe.msh"))
    mmesh = Mesh(mmesh.nodes / 5, mmesh.elems)
elseif pdata == 3
    mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))
    mmesh = Mesh(mmesh.nodes * 10, mmesh.elems)
end
