include("utils.jl")
include("earthquick_main.jl")

disp, state, v_bd = simulate()
disp0 = matread("disp.mat")["D"]
state0 = matread("disp.mat")["S"]
v_bd0 = matread("disp.mat")["B"]
loss = sum((disp-disp0)^2) + sum((state-state0)^2)+ sum((v_bd-v_bd0)^2)

sess = Session(); init(sess)

@show run(sess, loss)

# BFGS!(sess, loss)



# lineview(sess, a, loss, ones(n+1) * 0.01, ones(n+1) * 0.02)
# lineview(sess, b, loss,ones(n+1) * 0.02, ones(n+1) * 0.03)
# lineview(sess, σn, loss, ones(n+1) * 50.0, ones(n+1) * 30.0)
# lineview(sess, Dc, loss, 0.03, 1.0)
# lineview(sess, f0, loss, 0.6, 0.8)
# lineview(sess, v0, loss, 1e-6, 1e-4)
# lineview(sess, η, loss, 4.7434, 10.0)

# gradview(sess, a, loss,ones(n+1) * 0.02)
# gradview(sess, b, loss,ones(n+1) * 0.03)
# gradview(sess, σn, loss, ones(n+1) * 30.0)
# gradview(sess, Dc, loss, 1.0)
# gradview(sess, f0, loss, 0.8)
# gradview(sess, v0, loss, 1e-4)
# gradview(sess, η, loss, 10.0)

# gradview(sess, pl, loss, [1.2])

# # run(sess, gradients(loss, σn), b=>0.018*ones(n+1))

# disp0, state0, v_bd0 = run(sess, [disp, state, v_bd])
# matwrite("disp.mat", Dict("D"=>disp0, "S"=>state0, "B"=>v_bd0))
# gradients(loss, a)
# gradients(loss, b)
# gradients(loss, σn)
# gradients(loss, v0)
# gradients(loss, Dc)
# gradients(loss, η)
# gradients(loss, f0)

