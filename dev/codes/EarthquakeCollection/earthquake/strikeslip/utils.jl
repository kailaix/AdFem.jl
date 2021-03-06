using Revise
using ADCME
using AdFem
using SparseArrays
using PyPlot


function plot_disp(u, m, n, h)
    close("all")
    u = reshape(u, m+1, n+1)
    x = collect((0:m) * h)
    y = collect((0:n) * h)
    pcolormesh(x, y, u')
    colorbar()
    axis("scaled")
end

function plot_disp_line(u, m, n, h)
   close("all")
    u = reshape(u, m+1, n+1)
   plot(u[:, n÷2], label="middle")
   plot(u[:, n÷3], label="upper")
   plot(u[:, n÷3*2], label="lower")
   legend()
end
