using DelimitedFiles
using LinearAlgebra
using Printf 
using Latexify 
using Statistics
using PyPlot

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


res = readdlm("linear.txt")
d = Dict()
for i = 1:size(res,1)
    r = norm(reshape(res[i,2:end],3,3)-D)/norm(D)
    if haskey(d, res[i,1])
        push!(d[res[i,1]], r)
    else
        d[res[i,1]] = [r]
    end
end
arr = [0.0, 0.01, 0.05, 0.1]
s = zeros(length(arr))
m = zeros(length(arr))
for (ix,k) in enumerate(arr)
    s[ix] = std(d[k])
    m[ix] = mean(d[k])
end

semilogy(arr, m)
fill_between(arr, m - s, m+s, alpha=0.5)
xlabel("\$\\sigma_{\\mathrm{noise}}\$")
ylabel("Error")
savefig("test")
# mpl.save("poroerror.tex")



res = readdlm("visco.txt")
d = Dict()
for i = 1:size(res,1)
    r = (res[i,2] - D1[1])/D1[1]
    if haskey(d, res[i,1])
        push!(d[res[i,1]], r)
    else
        d[res[i,1]] = [r]
    end
end
arr = [0.0, 0.01, 0.05, 0.1]
s = zeros(length(arr))
m = zeros(length(arr))
for (ix,k) in enumerate(arr)
    s[ix] = std(d[k])
    m[ix] = mean(d[k])
end
close("all")
semilogy(arr, m)
fill_between(arr, m - s, m+s, alpha=0.5)
xlabel("\$\\sigma_{\\mathrm{noise}}\$")
ylabel("Error")
savefig("test")
# mpl.save("poroerror.tex")