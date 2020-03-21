using PyCall
using MAT
using PyPlot 
using Statistics
mpl = pyimport("tikzplotlib")

n = 15
m = 2n 
h = 0.01
NT = 20
Δt = 2/NT


function plot1(model_id, k)
    U0 = matread("data$model_id.mat")["U"][4]
    Sigma0 = matread("data$model_id.mat")["S"][4]

    d = matread("model$(model_id)nn$k.mat")
    Sigma_ = d["S"]
    U_ = d["U"]
    close("all")
    plot(LinRange(0, 2.0, NT+1), U0[:,1], "r--", label="\$u_x\$")  # reference
    plot(LinRange(0, 2.0, NT+1), U_[:,1], "ro")
    plot(LinRange(0, 2.0, NT+1), U0[:,1+(n+1)*(m+1)], "g--", label="\$u_y\$")
    plot(LinRange(0, 2.0, NT+1), U_[:,1+(n+1)*(m+1)], "go")
    xlabel("Time")
    ylabel("Displacement")
    legend()
    mpl.save("nlv$(model_id)_disp$k.tex")
    savefig("nlv$(model_id)_disp$k.png")

    close("all")
    plot(LinRange(0, 2.0, NT+1), mean(Sigma0[:,1:4,1], dims=2)[:],"r--", label="\$\\sigma_{xx}\$")
    plot(LinRange(0, 2.0, NT+1), mean(Sigma0[:,1:4,2], dims=2)[:],"b--", label="\$\\sigma_{yy}\$")
    plot(LinRange(0, 2.0, NT+1), mean(Sigma0[:,1:4,3], dims=2)[:],"g--", label="\$\\sigma_{xy}\$")
    plot(LinRange(0, 2.0, NT+1), mean(Sigma_[:,1:4,1], dims=2)[:],"ro")
    plot(LinRange(0, 2.0, NT+1), mean(Sigma_[:,1:4,2], dims=2)[:],"bo")
    plot(LinRange(0, 2.0, NT+1), mean(Sigma_[:,1:4,3], dims=2)[:],"go")
    legend()
    legend()
    xlabel("Time")
    ylabel("Stress")
    mpl.save("nlv$(model_id)_stress$k.tex")
    savefig("nlv$(model_id)_stress$k.png")
end

plot1(1, 0)
plot1(1, 9)
plot1(2, 0)
plot1(2, 9)