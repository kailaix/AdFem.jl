using AdFem
using Test
using PyPlot
using PyCall
using LinearAlgebra

np = pyimport("numpy")
m = 40
n = 20
h = 0.1
bdnode = Int64[]
bdval1 = Float64[]
bdval2 = Float64[]
X = (0:m)*h
Y = (0:n)*h
X, Y = np.meshgrid(X,Y)
for i = 1:m+1
    for j = 1:n+1
        push!(bdnode, i+(j-1)*(n+1))
        x = (i-1)*h
        y = (j-1)*h
        push!(bdval1, x^2+y^2)
        push!(bdval2, x^2-y^2)
    end
end
bdnode = [bdnode;bdnode.+(m+1)*(n+1)]
bdval = [bdval1;bdval2]

Xv = []
Yv = []
for j = 1:n 
    for i = 1:m 
        push!(Xv, (i-1/2)*h)
        push!(Yv, (j-1/2)*h)
    end
end

include("Core.jl")