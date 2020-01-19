@testset "compute_fem_stiffness_matrix 1" begin
    K = diagm(0=>ones(3))
    m = 40
    n = 20
    h = 0.1
    bdnode = Int64[]
    bdval1 = Float64[]
    bdval2 = Float64[]
    X = (0:n)*h
    Y = (0:m)*h
    X, Y = np.meshgrid(X,Y)
    for i = 1:n+1
        for j = 1:m+1
            push!(bdnode, i+(j-1)*(n+1))
            x = (i-1)*h
            y = (j-1)*h
            push!(bdval1, x^2+y^2)
            push!(bdval2, x^2-y^2)
        end
    end
    bdnode = [bdnode;bdnode.+(m+1)*(n+1)]
    bdval = [bdval1;bdval2]
    A = compute_fem_stiffness_matrix(K, m, n, h)
    rhs = compute_fem_source_term(3*ones(4m*n), -1*ones(4m*n), m, n, h)
    rhs[bdnode] = bdval
    At = trim_fem(A, bdnode, m, n, h)
    u = At\rhs 
    u1 = reshape(u[1:(m+1)*(n+1)], n+1, m+1)'|>Array
    u2 = reshape(u[(m+1)*(n+1)+1:end], n+1, m+1)'|>Array
    close("all"); 
    mesh(X,Y, u1); 
    mesh(X,Y, (@. X^2+Y^2), color="orange", alpha=0.5)
    savefig("compute_fem_stiffness_matrix1_u1.png")
    close("all"); 
    mesh(X,Y, u2); 
    mesh(X,Y, (@. X^2-Y^2), color="orange", alpha=0.5)
    savefig("compute_fem_stiffness_matrix1_u2.png")
end

@testset "compute_fem_stiffness_matrix 2" begin
    E = 1.0
    ν = 0.3

    K = E/(1-ν^2)*[
        1 ν 0.0
        ν 1.0 0.0
        0.0 0.0 (1-ν)/2
    ]
    m = 40
    n = 20
    h = 0.1
    bdnode = Int64[]
    bdval1 = Float64[]
    bdval2 = Float64[]
    X = (0:n)*h
    Y = (0:m)*h
    X, Y = np.meshgrid(X,Y)
    for i = 1:n+1
        for j = 1:m+1
            push!(bdnode, i+(j-1)*(n+1))
            x = (i-1)*h
            y = (j-1)*h
            push!(bdval1, x^2+y^2)
            push!(bdval2, x^2-y^2)
        end
    end
    bdnode = [bdnode;bdnode.+(m+1)*(n+1)]
    bdval = [bdval1;bdval2]
    A = compute_fem_stiffness_matrix(K, m, n, h)
    rhs = compute_fem_source_term(E/(1-ν^2)*(5-ν)/2*ones(4m*n), -E/(1-ν^2)*(ν+2)/2*ones(4m*n), m, n, h)
    rhs[bdnode] = bdval
    At = trim_fem(A, bdnode, m, n, h)
    u = At\rhs 
    u1 = reshape(u[1:(m+1)*(n+1)], n+1, m+1)'|>Array
    u2 = reshape(u[(m+1)*(n+1)+1:end], n+1, m+1)'|>Array
    close("all"); 
    mesh(X,Y, u1); 
    mesh(X,Y, (@. X^2+Y^2), color="orange", alpha=0.5)
    savefig("compute_fem_stiffness_matrix2_u1.png")
    close("all"); 
    mesh(X,Y, u2); 
    mesh(X,Y, (@. X^2-Y^2), color="orange", alpha=0.5)
    savefig("compute_fem_stiffness_matrix2_u2.png")
end


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

@testset "compute_fem_normal_traction_term" begin
    bdedge = []
    for i = 1:m 
        push!(bdedge, [i i+1])
        push!(bdedge, [n*(m+1)+i n*(m+1)+i+1])
    end
    for j = 1:n 
        push!(bdedge, [(j-1)*(m+1)+1 j*(m+1)+1])
        push!(bdedge, [(j-1)*(m+1)+m+1 j*(m+1)+m+1])
    end
    bdedge = vcat(bdedge...)
    rhs = compute_fem_normal_traction_term(10.0, bdedge, m, n, h)
    @test sum(rhs)≈120
end