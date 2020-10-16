@testset "eval_strain_on_gauss_pts1" begin
    m = 20
    n = 10
    h = 0.1
    u3 = zeros((m+1)*(n+1))
    a = zeros(m, n)
    b = zeros(m, n)
    for i = 1:m+1
        for j = 1:n+1
            x = (i-1)*h 
            y = (j-1)*h 
            u3[ i+(j-1)*(m+1) ] = x^2+y^2
            if i<=m && j<=n
                a[i, j] = 2x
                b[i,j] = 2y
            end
        end
    end
    s = eval_strain_on_gauss_pts1(constant(u3), m, n, h)

    S = run(sess, s)
    A = reshape(S[1:4:end,1], m, n)
    B = reshape(S[1:4:end,2], m, n)

    close("all")
    mesh(A)
    mesh(a, color="orange")

    close("all")
    mesh(B)
    mesh(b, color="orange")




end


@testset "compute_fem_stiffness_matrix1" begin
using Revise
using AdFem
using ADCME
    m = 10
    n = 5
    h = 0.1
    U = zeros((m+1)*(n+1))
    for i = 1:m+1
        for j = 1:n+1
            x = (i-1)*h 
            y = (j-1)*h 
            U[i+(j-1)*(m+1)] = x^2 + y^2
        end
    end
    
    hmat = zeros(4m*n, 2, 2)
    for i = 1:4*m*n
        hmat[i,:,:] = [2. 1.;3. 4.]
    end
    A = compute_fem_stiffness_matrix1(constant(hmat), m, n, h)
    A2 = compute_fem_stiffness_matrix1([2. 1.;3. 4.], m, n, h)
    sess = Session()
    run(sess, A) - A2
    
end