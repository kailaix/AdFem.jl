using PoreFlow

m = 100; n = 100; h = 1/n
mesh = Mesh(m, n, h)

function f(x, y)
    2 * x^2 + y
end

f_fem = eval_f_on_fem_pts(f, mesh)
f_gauss = fem_to_gauss_points(f_fem, mesh)
sess = Session(); init(sess)
f_gauss = run(sess, f_gauss)

f_gauss2 = eval_f_on_gauss_pts(f, mesh)

maximum(abs.(f_gauss .- f_gauss2))