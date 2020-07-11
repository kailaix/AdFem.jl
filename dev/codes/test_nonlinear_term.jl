using PoreFlow
m = 5
n = 5
h = 1/n

# F(u + Delta u, v) = F_u Delta u + F 

# F(u , v + Delta v) = F + F_v Delta v

function eval_f(S)
    u, v = S[1:(m+1)*(n+1)], S[(m+1)*(n+1)+1:2(m+1)*(n+1)]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    # ux = ones(4*m*n)

    f1 = compute_fem_source_term1(ugauss.*ux, m, n, h)
    f2 = compute_fem_source_term1(vgauss.*uy, m, n, h)
    return f1 + f2
end

function eval_df(S)
    u, v = S[1:(m+1)*(n+1)], S[(m+1)*(n+1)+1:2(m+1)*(n+1)]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    M1 = compute_fem_mass_matrix1(ux, m, n, h)
    M2 = run(sess, compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h)) # a julia kernel needed
    Fv = compute_fem_mass_matrix1(uy, m, n, h)
    return M1+M2, Fv
end

sess = Session()
S = rand(2(m+1)*(n+1))
function fJ(x)
    global S
    S[(m+1)*(n+1)+1:end] = x
    f = eval_f(S)
    _, df = eval_df(S)
    return f, df
end

function fJ2(x)
    global S
    S[1:(m+1)*(n+1)] = x
    f = eval_f(S)
    df, _ = eval_df(S)
    return f, df
end

function ADCME.:test_jacobian(f::Function, x0::Array{Float64}; scale::Float64 = 1.0)
    v0 = rand(Float64,size(x0))
    γs = scale ./10 .^(1:5)
    err2 = []
    err1 = []
    f0, J = f(x0)
    for i = 1:5
        f1, _ = f(x0+γs[i]*v0)
        push!(err1, norm(f1-f0))
        @show f1, f0, 2γs[i]*J*v0
        push!(err2, norm(f1-f0-γs[i]*J*v0))
        # push!(err2, norm((f1-f2)/(2γs[i])-J*v0))
        # #@show "test ", f1, f2, f1-f2
    end
    close("all")
    loglog(γs, err2, label="Automatic Differentiation")
    loglog(γs, err1, label="Finite Difference")
    loglog(γs, γs.^2 * 0.5*abs(err2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(γs, γs * 0.5*abs(err1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
    legend()
    println("Finite difference: $err1")
    println("Automatic differentiation: $err2")
    return err1, err2
end

u0 = rand((m+1)*(n+1))
fJ(u0)

close("all")
test_jacobian(fJ2, u0)