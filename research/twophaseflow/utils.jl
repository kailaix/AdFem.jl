using Revise
using FwiFlow
using PoreFlow
using ADCME
using PyPlot
using PyCall
using Statistics 
using ADCMEKit

function visualize_saturation(s2)
    # fig, ax = subplots()
    close("all")
    ln = pcolormesh(s2[1,:,:], vmin=0.0, vmax=1.0)
    t = title("t = 0")
    colorbar()
    axis("scaled")
    function update(frame)
        t.set_text("t = $(round(frame * Δt, digits=3))")
        ln.set_array(s2[frame,:,:]'[:])
    end
    anim = animate(update, 1:size(s2,1))
end

function visualize_potential(φ)
    m = mean(φ)
    s = std(φ)
    close("all")
    vmin, vmax = m - 2s, m + 2s
    ln = pcolormesh(φ[1,:,:], vmin= vmin, vmax=vmax)
    colorbar()
    c = contour(φ[1,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
    t = title("t = 0")
    axis("scaled")
    function update(i)
        gca().clear()
        # t.set_text("t = $(round(frame * Δt, digits=3))")
        # ln.set_array(φ[frame,:,:]'[:])
        # c.set_array(φ[frame,:,:]'[:])
        ln = gca().pcolormesh(φ[i,:,:], vmin= vmin, vmax=vmax)
        c = gca().contour(φ[i,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
        t = gca().set_title("t = $i")
    end
    anim = animate(update, 1:size(φ,1))
end

function visualize_displacement(u)
    X = zeros(m+1, n+1)
    Y = zeros(m+1, n+1)
    for i = 1:m+1
        for j = 1:n+1
            X[i, j] = (i-1)*h 
            Y[i, j] = (j-1)*h 
        end
    end
    function disp(u)
        U1 = reshape(u[1:(m+1)*(n+1)], m+1, n+1)
        U2 = reshape(u[(m+1)*(n+1)+1:end], m+1, n+1)
        U1 = X + U1 
        U2 = Y + U2
        U1, U2 
    end
    close("all")
    U1, U2 = disp(u[1,:])
    s = scatter(U1[:], U2[:], s=1)
    xlim(-10h, 10h+(m+1)*h)
    ylim(-10h, 10h+(n+1)*h)
    gca().invert_yaxis()
    t = title("t = 0")
    function update(i)
        U1, U2 = disp(u[i,:])
        s.set_offsets([U1[:] U2[:]])
        t.set_text("t = $i")
    end
    animate(update, 1:size(u,1))
end