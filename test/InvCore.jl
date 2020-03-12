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