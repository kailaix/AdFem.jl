@testset "get_bdedge_integration_pts" begin 
    mesh = Mesh(1, 1, 1.0)
    bd = bcedge(mesh)
    pts = get_bdedge_integration_pts(bd, mesh)
    @test maximum(abs.(pts - [ 0.788675  0.0
            0.211325  0.0
            1.0       0.788675
            1.0       0.211325
            0.0       0.788675
            0.0       0.211325
            0.788675  1.0
            0.211325  1.0])) < 1e-5
end

@testset "traction" begin 

    for d in [1, 2]
        n = 20
        mesh = Mesh(n, n, 1/n; degree=d)
        bd = bcedge(mesh)

        # lower 
        bdedge = []
        for i = 1:size(bd, 1)
            a, b = bd[i,:]
            if mesh.nodes[a,2]<1e-5 && mesh.nodes[b, 2]<1e-5
                push!(bdedge, [a b])
            end
        end
        bdedge = vcat(bdedge...)
        t = eval_f_on_boundary_edge((x, y)->y^2 + x^2 + 1.0, bdedge, mesh)
        rhs = compute_fem_traction_term1(t, bdedge, mesh)
        @test abs(sum(rhs)-4/3)<1e-5


        # upper 
        bdedge = []
        for i = 1:size(bd, 1)
            a, b = bd[i,:]
            if mesh.nodes[a,2]>1-1e-5 && mesh.nodes[b, 2]>1-1e-5
                push!(bdedge, [a b])
            end
        end
        bdedge = vcat(bdedge...)
        t = eval_f_on_boundary_edge((x, y)-> y^2 + x^2 + 1.0, bdedge, mesh; order=6)
        rhs = compute_fem_traction_term1(t, bdedge, mesh; order=6)
        @test abs(sum(rhs)-7/3)<1e-5


        # left 
        bdedge = []
        for i = 1:size(bd, 1)
            a, b = bd[i,:]
            if mesh.nodes[a,1]<1e-5 && mesh.nodes[b, 1]<1e-5
                push!(bdedge, [a b])
            end
        end
        bdedge = vcat(bdedge...)
        t = eval_f_on_boundary_edge((x, y)->y^2 + x^2 + 1.0, bdedge, mesh)
        rhs = compute_fem_traction_term1(t, bdedge, mesh)
        @test abs(sum(rhs)-4/3)<1e-5

        # right 
        bdedge = []
        for i = 1:size(bd, 1)
            a, b = bd[i,:]
            if mesh.nodes[a,1]>1-1e-5 && mesh.nodes[b, 1]>1-1e-5
                push!(bdedge, [a b])
            end
        end
        bdedge = vcat(bdedge...)
        t = eval_f_on_boundary_edge((x, y)->y^2 + x^2 + 1.0, bdedge, mesh)
        rhs = compute_fem_traction_term1(t, bdedge, mesh)
        @test abs(sum(rhs)-7/3)<1e-5
    end
end

@testset "impose_Dirichlet_boundary_conditions" begin 
    A = rand(5, 5)
    bd = [5;2;3]
    bdval = Float64[1;2;3]
    rhs = rand(5)
    A1, rhs1 = impose_Dirichlet_boundary_conditions(A, rhs, bd, bdval)
    A2, rhs2 = impose_Dirichlet_boundary_conditions(constant(sparse(A)), rhs, bd, bdval)

    @test maximum(abs.(Array(run(sess, A2)) - A1)) < 1e-5
    @test maximum(abs.(run(sess, rhs2) - rhs1)) < 1e-5
end