@testset "get_edge_normal" begin 
    mmesh = Mesh(2,2,1.0)
    out = get_edge_normal(mmesh)
    bdedge = bcedge(mmesh)
    n1 = mmesh.nodes[bdedge[:,1], :]
    n2 = mmesh.nodes[bdedge[:,2], :]
    @test out≈[ -0.0  -1.0
        0.0   1.0
        0.0   1.0
        1.0   0.0
    -1.0  -0.0
    -1.0  -0.0
    -0.0  -1.0
        1.0   0.0]
end