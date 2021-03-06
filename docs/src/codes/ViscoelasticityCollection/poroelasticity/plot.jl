using PyPlot
using PyCall
using DelimitedFiles
using Statistics 
mpl = pyimport("tikzplotlib")

res = readdlm("result.txt")
d = Dict()
for i = 1:size(res,1)
    if haskey(d, res[i,1])
        push!(d[res[i,1]], res[i,2])
    else
        d[res[i,1]] = [res[i,2]]
    end
end


arr = [0.0,0.002,0.004,0.006,0.008,0.01,0.03,0.05,0.06, 0.07, 0.075,0.08,0.09,0.1]
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
mpl.save("poroerror.tex")