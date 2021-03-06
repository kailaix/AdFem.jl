using Revise
using FwiFlow
using AdFem
using ADCME
using PyPlot
using PyCall
using Statistics 
using DelimitedFiles
using ADCMEKit
np = pyimport("numpy")

function visualize_obs(o1)
    close("all")
    plot((0:NT)*Δt, o1[:, 1:m+1])
    xlabel("t")
    ylabel("displacement")
end

function layer_model(m, n)
    z = zeros(4*n*m)
    k = 0
    mm = ones(n)
    mm[n÷3:end] .= 1.5
    for i = 1:n 
        for j = 1:m 
            for p = 1:2
                for q = 1:2
                    k += 1
                    z[k] = mm[i]
                end
            end
        end
    end
    return z
end

function sin_model(m, n)
end

function visualize_invη(o)
    o = o * 1e12
    Z = zeros(n, m)
    k = 0
    for i = 1:n 
        for j = 1:m 
            Z[i, j] = mean(o[k+1:k+4])
            k += 4
        end
    end
    pcolormesh(Z)
    axis("scaled")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    colorbar()
end