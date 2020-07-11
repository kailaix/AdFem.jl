@testset "compute_fvm_tpfa_matrix" begin 
    m = 20
    n = 10
    h = 0.1
    kvalue = rand(m*n)
    bc = bcedge("all", m, n, h)
    pval = rand(size(bc, 1))
    K, rhs = compute_fvm_tpfa_matrix(kvalue,bc,pval,m,n,h)
    K1, rhs1 = compute_fvm_tpfa_matrix(constant(kvalue),bc,pval,m,n,h)
    K2 = run(sess, K1)
    rhs2 = run(sess, rhs1)
    @test rhs≈rhs2
    @test K ≈ K2

    m = 20
    n = 10
    h = 0.1
    kvalue = rand(m*n)
    K = compute_fvm_tpfa_matrix(kvalue,m,n,h)
    K1 = compute_fvm_tpfa_matrix(constant(kvalue),m,n,h)
    K2 = run(sess, K1)
    @test K ≈ K2
end



function advection_jl(m, n, h, v, u)
    # v: velocity, (m*n) * 2
    # u: temperature, (m*n)
    a = zeros(m*n)
    for j=1:n
        for i=1:m
            vx = v[(j-1)*m+i, 1]
            vy = v[(j-1)*m+i, 2]
            if vx >= 0
                if i==1
                    a[(j-1)*m+i] += vx * 2 * h * u[(j-1)*m+i]
                else
                    a[(j-1)*m+i] += vx * h * (u[(j-1)*m+i] - u[(j-1)*m+i-1])
                end
            else
                if i==m
                    a[(j-1)*m+i] += - vx * 2 * h * u[(j-1)*m+i]
                else
                    a[(j-1)*m+i] += vx * h * (u[(j-1)*m+i+1] - u[(j-1)*m+i])
                end
            end
                

            if vy >= 0
                if j==1
                    a[(j-1)*m+i] += vy * 2 * h * u[(j-1)*m+i]
                else
                    a[(j-1)*m+i] += vy * h * (u[(j-1)*m+i] - u[(j-2)*m+i])
                end
            else
                if j==n
                    a[(j-1)*m+i] += - vy * 2 * h * u[(j-1)*m+i]
                    
                else
                    a[(j-1)*m+i] += vy * h * (u[j*m+i] - u[(j-1)*m+i])
                end
            end
        end
    end
    return a
end


@testset "compute_fvm_advection_term" begin 
    m = 10
    n = 10
    h = 0.1
    v = rand(m*n,2)
    u = rand(m*n)
    jl_u = advection_jl(m, n, h, v, u)
    u = compute_fvm_advection_term(v,u,m,n,h)
    @show run(sess, u)≈jl_u
end


@testset "grad on gauss points" begin 
    m = 10
    n = 10
    h = 0.1
    u = rand(2*(m+1)*(n+1))
    ref = eval_grad_on_gauss_pts(u, m, n, h)
    out = eval_grad_on_gauss_pts(constant(u), m, n, h)
    @test run(sess, out)≈ref
end

@testset "fem_to_gauss_points2" begin 
    m = 10
    n = 10
    h = 0.1
    xy = fem_nodes(m, n, h)
    x, y = xy[:,1], xy[:,2]
    u0 = @. x*y*(1-x^2)
    v0 = fem_to_gauss_points(u0, m, n, h)
    @test run(sess, fem_to_gauss_points(constant(u0), m, n, h)) ≈ v0

    figure(figsize=(10,4))
    subplot(121)
    visualize_scalar_on_fem_points(u0, m, n, h)
    subplot(122)
    visualize_scalar_on_gauss_points(v0, m, n, h)
end

@testset "eval_grad_on_gauss_pts" begin 
    m = 10
    n = 10
    h = 0.1
    xy = fem_nodes(m, n, h)
    x, y = xy[:,1], xy[:,2]
    u0 = @. x*y*(1-x^2)
    u1 = @. sin(x) * cos(y)
    G = eval_grad_on_gauss_pts([u0;u1], m, n, h)

    xy = gauss_nodes(m, n, h)
    x, y = xy[:,1], xy[:,2]
    g11 = @. y-3x^2*y
    g12 = @. x-x^3
    g21 = @. cos(x) * cos(y)
    g22 = @. -sin(x) * sin(y)
    
    figure(figsize=(10,10))
    subplot(221)
    visualize_scalar_on_gauss_points(G[:,1,1], m, n, h)
    subplot(222)
    visualize_scalar_on_gauss_points(G[:,1,2], m, n, h)
    subplot(223)
    visualize_scalar_on_gauss_points(G[:,2,1], m, n, h)
    subplot(224)
    visualize_scalar_on_gauss_points(G[:,2,2], m, n, h)

    figure(figsize=(10,10))
    subplot(221)
    visualize_scalar_on_gauss_points(g11, m, n, h)
    subplot(222)
    visualize_scalar_on_gauss_points(g12, m, n, h)
    subplot(223)
    visualize_scalar_on_gauss_points(g21, m, n, h)
    subplot(224)
    visualize_scalar_on_gauss_points(g22, m, n, h)
end


