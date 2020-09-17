using PoreFlow

m = 100; n = 100; h = 1/n
mesh = Mesh(m, n, h)

function f(x, y)
    2 * x^2 + x + 14 * y
end

function fx(x, y)
    4 * x + 1
end

function fy(x, y)
    14.0
end

f_fem = eval_f_on_fem_pts(f, mesh)
f_grad = eval_grad_on_gauss_pts1(f_fem, mesh)
sess = Session(); init(sess)
f_grad = run(sess, f_grad)
fx1, fy1 = f_grad[:, 1], f_grad[:, 2]

fx2 = eval_f_on_gauss_pts(fx, mesh)
fy2 = eval_f_on_gauss_pts(fy, mesh)

maximum(abs.(fx1 .- fx2))
maximum(abs.(fy1 .- fy2))