# matplotlib.use("macosx")

@testset "eval_f_on_gauss_pts" begin
    f = (x,y)->x^2+y^2
    F = eval_f_on_gauss_pts(f, m, n, h)
    pts = PoreFlow.get_gauss_points(m, n, h)
    close("all")
    scatter3D(pts[:,1], pts[:,2], F, marker=".")
    mesh(X, Y, (@. f(X, Y)), color="orange", alpha=0.5)
    savefig("eval_f_on_gauss_pts.png")
end

@testset "compute_fvm_mechanics_term" begin
    u = zeros(2*(m+1)*(n+1))
    k = 0
    for j = 1:n+1
        for i = 1:m+1    
            x = (i-1)*h
            y = (j-1)*h
            k += 1
            u[k] = x^2+y^2
            u[k + (m+1)*(n+1)] = x^2-y^2
        end
    end
    eps_v = compute_fvm_mechanics_term(u, m, n, h)
    close("all")
    scatter3D(Xv, Yv, eps_v, marker=".")
    scatter3D(Xv, Yv, 2h^2 * (@. Xv-Yv), marker="+")
    savefig("compute_fvm_mechanics_term.png")
end

@testset "compute_interaction_matrix" begin
    A = compute_interaction_matrix(m, n, h)
    u = zeros(2*(m+1)*(n+1))
    k = 0
    for j = 1:n+1
        for i = 1:m+1    
            x = (i-1)*h
            y = (j-1)*h
            k += 1
            u[k] = x^2+y^2
            u[k + (m+1)*(n+1)] = x^2-y^2
        end
    end
    p = zeros(m*n)
    k = 0
    for j = 1:n 
        for i = 1:m    
            k += 1
            x = (i-1/2)*h
            y = (j-1/2)*h
            p[k] =  x+y
        end
    end
    val = p'*A*u 
    @test val ≈ 64
end

@testset "compute_fvm_source_term" begin
    f = (x,y)->x^2+y^2
    F = eval_f_on_gauss_pts(f, m, n, h)
    S = compute_fvm_source_term(F, m, n, h)
    Fs = f.(Xv, Yv)*h^2
    close("all")
    scatter3D(Xv, Yv, S, marker = ".") 
    scatter3D(Xv, Yv, Fs, marker = "+") 
    savefig("compute_fvm_source_term.png")
end

@testset "compute_fluid_tpfa_matrix" begin
    M = compute_fluid_tpfa_matrix(m, n, h)
    p = Xv.^2 + Yv.^2
    V = M * p 
    close("all")
    scatter3D(Xv, Yv, V, marker = ".") 
    scatter3D(Xv, Yv, 4h^2 * ones(m*n), marker = "+") 
    savefig("compute_fluid_tpfa_matrix.png")
end

@testset "compute_fem_stiffness_matrix compute_fem_traction_term" begin
    # bdedge = []
    # for i = 1:m 
    #     push!(bdedge, [i i+1])
    # end
    # bdedge = vcat(bdedge...)

    bdnode = Int64[]
    for j = 1:n+1
        push!(bdnode, (j-1)*(m+1)+1)
        push!(bdnode, (j-1)*(m+1)+m+1)
    end
    for i = 2:m
        push!(bdnode, n*(m+1)+i)
        push!(bdnode, i)
    end

    F1 = eval_f_on_gauss_pts((x,y)->3.0, m, n, h)
    F2 = eval_f_on_gauss_pts((x,y)->-1.0, m, n, h)
    F = compute_fem_source_term(F1, F2, m, n, h)

    # t1 = eval_f_on_boundary_edge((x,y)->2x, bdedge, m, n, h)
    # t2 = eval_f_on_boundary_edge((x,y)->x+y, bdedge, m, n, h)
    # T = compute_fem_traction_term([t1 t2], bdedge, m, n, h)
    
    D = diagm(0=>[1,1,0.5])
    K = compute_fem_stiffness_matrix(D, m, n, h)
    rhs =  - F 
    bdval = [eval_f_on_boundary_node((x,y)->x^2+y^2, bdnode, m, n, h);
            eval_f_on_boundary_node((x,y)->x^2-y^2, bdnode, m, n, h)]
    rhs[[bdnode;bdnode .+ (m+1)*(n+1)]] = bdval
    K, res = fem_impose_Dirichlet_boundary_condition(K, bdnode, m, n, h; bdval = bdval)
    u = K\(rhs+res)
    X, Y, U, V = plot_u(u, m, n, h)

    mesh(X, Y, (@. X^2+Y^2-U), alpha=0.6, color="blue")
    mesh(X, Y, (@. X^2-Y^2-V), alpha=0.6, color="orange")

    @test sum(rhs)≈120
end