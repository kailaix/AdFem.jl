using DelimitedFiles
using LinearAlgebra
using Printf 
using Latexify 
using Statistics
using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")

E = 6.e9
ν = 0.35
D = E/(1+ν)/(1-2ν)*[1-ν ν 0;ν 1-ν 0;0 0 (1-2ν)/2] 

μ = E/(2(1+ν))
λ = E*ν/(1+ν)/(1-2ν)
D2 = [μ λ 1.5e-12]
D1 = D2 
# for noise in [0.0, 0.01, 0.05, 0.1]
#     D1 = readdlm("linear$noise.txt")
#     @printf "%0.4e\n" norm(D1-D)/norm(D)
#     println(latexify(round.(D1/1e9, digits=4)))
# end

# D2 = [μ λ 1.5e-12]
# for noise in [0.0, 0.01, 0.05, 0.1]
#     D1 = readdlm("visco$noise.txt")
#     # @show round.(D1 ./ [1e9 1e9 1e-12], digits=4)
#     e = abs.(D1-D2)./abs.(D2)
#     @printf "%0.2e, %0.2e, %0.2e\n" e[1] e[2] e[3]
#     # println(latexify(round.(D1/1e9, digits=4)))
# end


close("all")
res = readdlm("linear.txt")
d = Dict()
for i = 1:size(res,1)
    try
    r = norm(reshape(res[i,2:end],3,3)-D)/norm(D)
    if haskey(d, res[i,1])
        push!(d[res[i,1]], r)
    else
        d[res[i,1]] = [r]
    end
    catch
    end
end
arr = [0.0, 0.001,0.003,0.009,
        0.01, 0.02,0.03,0.04, 0.05,0.06, 0.1]
s = zeros(length(arr))
m = zeros(length(arr))
for (ix,k) in enumerate(arr)
    d[k] = sort(d[k])[2:end-1]
    s[ix] = std(d[k])
    m[ix] = mean(d[k])
end

semilogy(arr, m)
fill_between(arr, m - s, m+s, alpha=0.5)
xlabel("\$\\sigma_{\\mathrm{noise}}\$")
ylabel("Error")
# savefig("test")
mpl.save("twolinear.tex")

plotval = k -> begin 
    res = readdlm("visco.txt")
    d = Dict()
    for i = 1:size(res,1)
        try
            r = abs(res[i,k+1] - D1[k])/D1[k]
            if haskey(d, res[i,1])
                push!(d[res[i,1]], r)
            else
                d[res[i,1]] = [r]
            end
        catch
        end
    end
    s = zeros(length(arr))
    m = zeros(length(arr))
    for (ix,k) in enumerate(arr)
        d[k] = sort(d[k])[2:end-1]
        s[ix] = std(d[k])
        m[ix] = mean(d[k])
    end
    local l 
    if k == 1
        l = "\$\\mu\$"
    elseif k==2
        l = "\$\\lambda\$"
    elseif k==3
        l = "\$\\eta\$"
    end
    semilogy(arr, m, label=l)
    fill_between(arr, m - s, m+s, alpha=0.5)
end
close("all")
plotval(1)
plotval(2)
plotval(3)
xlabel("\$\\sigma_{\\mathrm{noise}}\$")
ylabel("Error")
legend()
savefig("test")
mpl.save("twovisco.tex")
