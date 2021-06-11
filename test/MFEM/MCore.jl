@testset "eval scalar" begin 
    mmesh = Mesh(
        [2.2 2
        3. 4.
        2.5 10.], [1 2 3]
    )
    edge = [1 2]
    out = eval_scalar_on_boundary_edge([5.0;6.0;10000.0], edge, mmesh)
    sess = Session(); init(sess)
    @test run(sess, out)≈[5.069431844202973
    5.330009478207572
    5.669990521792428
    5.930568155797027]
end 