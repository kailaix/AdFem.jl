using PyPlot
using PyCall
using MAT 
using LinearAlgebra
using Statistics
mpl = pyimport("tikzplotlib")

n = 10
m = 2n 
h = 0.01
out = 0.5 * ones(n)
out[1:div(n,3)] .= 1.0

out *= 50 
rg = 0:20:200
err = zeros(4, length(rg))
for (j,idx) in enumerate([1])
    for (i,k) in enumerate(rg)
        eta = matread("$idx/eta$k.mat")["eta"]
        eta = eta[1:4m:end]
        err[j, i] = mean( ((out-eta) ./ out).^2)
        @info j, i, err[j,i]
    end
end

close("all")
# tp = ["ro-", "g+-", "b*-", "k>-"]
# labels = []
# for k in [1,2,3,5]
# s = length(1:k:m+1)
# push!(labels, "$s Sensors")
# end 
# for i = 1:4
#     plot(rg .+ 1, err[i,:], tp[i], label=labels[i])
# end
semilogy(rg .+ 1, err[1,:], tp[1])
legend()
xlabel("Iterations")
ylabel("Error")
mpl.save("space_loss.tex")
savefig("loss.png")



function visualize_inv_eta(X)
    ηmin = 0.5*50.
    ηmax = 50.
    x = LinRange(0.5h,m*h, m)
    y = LinRange(0.5h,n*h, n)
    V = zeros(m, n)
    for i = 1:m  
        for j = 1:n 
            elem = (j-1)*m + i 
            V[i, j] = mean(X[4(elem-1)+1:4elem])
        end
    end
    close("all")
    pcolormesh(x, y, V', vmin=ηmin-(ηmax-ηmin)/4, vmax=ηmax+(ηmax-ηmin)/4, rasterized=true)
    colorbar(shrink=0.5)
    xlabel("x")
    ylabel("y")
    # title("Iteration = $k")
    axis("scaled")
    gca().invert_yaxis()
end


out = repeat(out, 1, 4m)'[:]
visualize_inv_eta(out)
savefig("space_true.pdf")

eta  = matread("1/eta0.mat")["eta"]
visualize_inv_eta(eta)
savefig("space0.pdf")


eta  = matread("1/eta60.mat")["eta"]
visualize_inv_eta(eta)
savefig("space60.pdf")

eta  = matread("1/eta120.mat")["eta"]
visualize_inv_eta(eta)
savefig("space120.pdf")
