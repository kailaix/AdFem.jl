include("utils.jl")
include("earthquake_main.jl")

disp, state, v_bd = simulate()

# mode = "data"
mode = "inv"

if mode == "data"
    
    init(sess)
    disp0, state0, v_bd0 = run(sess, [disp, state, v_bd]);
    matwrite("data-eq.mat", Dict("disp"=>disp0, "state"=>state0, "v_bd"=>v_bd0))

    close("all")
    figure(figsize=[8, 6])
    subplot(311)
    plot(t, v_bd0[1:end-1,1], "k", label="V")
    ylabel("Velosity (m/s)")
    legend()
    xlim([t[1], t[end]])
    # gca().set_ylim(bottom=1e-5)
    gca().get_xaxis().set_ticklabels([])
    subplot(312)
    plot(t, state0[1:end-1,1], "k", label=L"\Psi")
    ylabel("State Variable")
    legend()
    xlim([t[1], t[end]])
    gca().get_xaxis().set_ticklabels([])
    subplot(313)
    plot(t, disp0[1:end-1,1], "k", label="U")
    ylabel("Slip (earthquick_main)")
    legend()
    xlim([t[1], t[end]])
    xlabel("Time (s)")
    savefig("earthquake-simulation.png", bbox_inches="tight")

end

if mode == "inv"

    disp0 = matread("data-eq.mat")["disp"]
    state0 = matread("data-eq.mat")["state"]
    v_bd0 = matread("data-eq.mat")["v_bd"]
    loss = mean((disp[:,1:m+1]-disp0[:,1:m+1])^2) + 
            mean((state-state0)^2) + 
            mean((v_bd-v_bd0)^2)

    loss = 1e10*loss
    sess = Session(); init(sess)

    @show run(sess, loss)
    # BFGS!(sess, loss)

## DEBUG
# gradients(loss, a)
# gradients(loss, b)
# gradients(loss, σn)
# gradients(loss, v0)
# gradients(loss, Dc)
# gradients(loss, η)
# gradients(loss, f0)
# gradients(loss, bd_left0)

# lineview(sess, a, loss, ones(n+1) * 0.01, ones(n+1) * 0.02)
# lineview(sess, b, loss,ones(n+1) * 0.02, ones(n+1) * 0.03)
# lineview(sess, σn, loss, ones(n+1) * 50.0, ones(n+1) * 30.0)
# lineview(sess, Dc, loss, 0.03, 1.0)
# lineview(sess, f0, loss, 0.6, 0.8)
# lineview(sess, v0, loss, 1e-6, 1e-4)
# lineview(sess, η, loss, 4.7434, 10.0)
# lineview(sess, bd_left0, loss, ones(n+1) * 0.3, ones(n+1) * 0.4)

gradview(sess, a, loss, ones(n+1) * 0.02)
# gradview(sess, b, loss, ones(n+1) * 0.03)
# gradview(sess, σn, loss, ones(n+1) * 30.0)
# gradview(sess, Dc, loss, 1.0)
# gradview(sess, f0, loss, 0.8)
# gradview(sess, v0, loss, 1e-4)
# gradview(sess, η, loss, 10.0)
# gradview(sess, bd_left0, loss, ones(n+1) * 0.3)

# gradview(sess, pl, loss, [1.2])

# # run(sess, gradients(loss, σn), b=>0.018*ones(n+1)

end