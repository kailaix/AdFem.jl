using PoreFlow

m = 1; n =1; h = 1/n
mesh = Mesh(m, n, h)

function f(x, y)
    x^2 + y
end

f_fem = eval_f_on_fem_pts(f, mesh)
f_gauss = fem_to_gauss_points(f_fem, mesh)
sess = Session(); init(sess)
f_gauss = run(sess, f_gauss)

f_gauss2 = eval_f_on_gauss_pts(f, mesh)

maximum(f_gauss .- f_gauss2)
f_gauss .- f_gauss2