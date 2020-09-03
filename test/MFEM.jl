using Revise
using ADCME
using PyPlot 
using PoreFlow 
using Test

mesh = Mesh(2, 2, 0.5)
@test mesh.nodes≈[ 0.0  0.0
                0.5  0.0
                1.0  0.0
                0.0  0.5
                0.5  0.5
                1.0  0.5
                0.0  1.0
                0.5  1.0
                1.0  1.0]

@test mesh.elems≈[ 1  2  4
                2  4  5
                2  3  5
                3  5  6
                4  5  7
                5  7  8
                5  6  8
                6  8  9]

@test get_ngauss(mesh)≈24


nodes = gauss_nodes(mesh)
close("all")
visualize_mesh(mesh)
scatter(nodes[:,1], nodes[:,2])
savefig("g.png")

@test get_area(mesh)≈ones(8)*0.25/2