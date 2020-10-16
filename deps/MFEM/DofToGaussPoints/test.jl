using AdFem

for degree in [1, 2]
    m = 1; n = 1; h = 1/n
    mesh = Mesh(m, n, h, degree = degree)

    function f(x, y)
        2 * x ^ 2 + y
    end

    f_dof = eval_f_on_dof_pts(f, mesh)
    f_gauss = dof_to_gauss_points(constant(f_dof), mesh)
    f_gauss0 = dof_to_gauss_points(f_dof, mesh)
    sess = Session(); init(sess)
    f_gauss = run(sess, f_gauss)
    @show maximum(abs.(f_gauss0 - f_gauss))

    f_gauss2 = eval_f_on_gauss_pts(f, mesh)

    @show maximum(abs.(f_gauss .- f_gauss2))
end