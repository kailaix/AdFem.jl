using PyCall
using MAT
using PyPlot 
using Statistics
using ADCME
using DelimitedFiles
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

function plot2(model_id)
    m = 30
    n = 15
    θ = matread("nn$(model_id)_20.mat")["Variablecolon0"]
    U0 = matread("data$model_id.mat")["U"][4]
    Sigma0 = matread("data$model_id.mat")["S"][4]
    S = Sigma0[end,:,:]
    SS = zeros(1800÷4,3)
    for i = 1:450
        SS[i,:] = mean(S[4(i-1)+1:4i,:], dims=1)[:]
    end
    val = squeeze(ae(SS, [20,20,20,1], θ))
    σ = constant(SS)
    if model_id==1
        val2 = constant(10*ones(m*n)) + 5.0/(1+1000*sum(σ[:,1:2]^2, dims=2))
    else
        val2 = constant(10*ones(m*n)) + relu(50.0/(1+1000*sum(σ[:,1:2]^2, dims=2)) - 10.0)
    end
    sess = Session()
    V1, V2, S = run(sess, [val, val2, σ])

    close("all")
    plot((@. (S[:,1]^2 + S[:,2]^2)), V1, "o", label="Estimated")
    plot((@. (S[:,1]^2 + S[:,2]^2)), V2, "+", label="Reference")
    

    θ = matread("nn$(model_id)_0.mat")["Variablecolon0"]
    U0 = matread("data$model_id.mat")["U"][4]
    Sigma0 = matread("data$model_id.mat")["S"][4]
    S = Sigma0[end,:,:]
    SS = zeros(1800÷4,3)
    for i = 1:450
        SS[i,:] = mean(S[4(i-1)+1:4i,:], dims=1)[:]
    end
    val = squeeze(ae(SS, [20,20,20,1], θ))
    σ = constant(SS)
    val2 = constant(10*ones(m*n)) + 5.0/(1+1000*sum(σ[:,1:2]^2, dims=2))
    sess = Session()
    V1, V2, S = run(sess, [val, val2, σ])
    plot((@. (S[:,1]^2 + S[:,2]^2)), V1, ".", label = "Initial Guess")
    
    legend()
    xlabel("\$\\sqrt{\\sigma_{11}^2+\\sigma_{22}^2}\$")
    ylabel("\$\\eta\$")    
    mpl.save("nlv$(model_id).tex")
    savefig("nlv$(model_id).png")
end

function normalize(x, n=201)
    T = zeros(n)
    T[1:length(x)] = log.(x)/log(10)
    for i = length(x)+1:n 
        T[i] = T[length(x)]
    end
    return T
end

function get_data(model_id, width, depth, activation)
    loss = readdlm("data/loss_$(model_id)_$(width)_$(depth)_$(activation).txt")[:]
    return normalize(loss)
end


function plot3(model_id, depth)
    U = zeros(20, 201)
    close("all")
    k = 0
    for activation in ["tanh", "relu", "selu", "elu"]
        for width in [1 5 10 20 40]
            k += 1
            U[k,:] = get_data(model_id, width, depth, activation)
            @info activation, width
        end
    end
    # subplot(120+v)
    close("all")
    pcolormesh(U', rasterized=true, cmap="hot")
    xticks([collect(0.5:1.0:19.5);2.5;7.5;12.5;17.5], 
        [repeat([1 5 10 20 40], 1,4)[:];"tanh"; "relu"; "selu"; "elu"])
    c = colorbar()
    c.set_ticks([-2,-1,0,1,2,3])
    c.set_ticklabels(["\$10^{-2}\$", "\$10^{-1}\$", 
            "\$10^0\$", "\$10^{1}\$", "\$10^{2}\$", "\$10^{3}\$"])
    va = -0.05*ones(5)
    ts = gca().get_xticklabels( )
    for i = 21:24
        ts[i].set_y( va[i-20] )
    end
    z = 0:200
    for i = 0:20
        plot(ones(201)*i, z, "k", alpha=0.2)
    end
    for i in [5, 10, 15]
        plot(ones(201)*i, z, "k")
    end
    # gca().tick_params(axis="both", which="major", labelsize=20)
    gca().tick_params(axis="both", which="major", labelsize=10)
    ylabel("Iterations")

    savefig("nn$(model_id)$depth.png")
    savefig("nn$(model_id)$depth.pdf")
end

plot1(1, 0)
plot1(1, 20)
plot1(2, 0)
plot1(2, 20)

plot2(1)
plot2(2)

for i in [1,3,5,10]
    plot3(1, i)
    plot3(2, i)
end
