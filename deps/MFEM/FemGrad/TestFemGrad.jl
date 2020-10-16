using AdFem

m = 100; n = 100; h = 1/n
mesh = Mesh(m, n, h, degree=2)

function f(x, y)
    2 * x + x + 14 * y
end

function fx(x, y)
    3 
end

function fy(x, y)
    14.0
end

f_dof = eval_f_on_dof_pts(f, mesh)
f_grad = eval_grad_on_gauss_pts1(f_dof, mesh)
sess = Session(); init(sess)
f_grad = run(sess, f_grad)
fx1, fy1 = f_grad[:, 1], f_grad[:, 2]

fx2 = eval_f_on_gauss_pts(fx, mesh)
fy2 = eval_f_on_gauss_pts(fy, mesh)

@info maximum(abs.(fx1 .- fx2))
@info maximum(abs.(fy1 .- fy2))