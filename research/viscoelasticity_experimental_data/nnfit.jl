using MAT
using PyPlot
using ADCME
using DelimitedFiles


data = matread("data.mat")["data"]

kid = 3
tid = 1

if length(ARGS)==2
    global kid = parse(Int64, ARGS[1])
    global tid = parse(Int64, ARGS[2])
end

@info kid, tid

d = data[kid]
n = div(size(d, 1), 5)

H = Variable(1.0)^2
ε = d[1:n, 1]
σ = d[1:n, 2]

function prediction(ε, σ0)
    ε, σ0 = convert_to_tensor([ε, σ0], [Float64, Float64])
    N = length(ε)
    condition = (i, σ)-> (i<=N)
    function body(i, taσ)
        σl = read(taσ, i-1)
        εc = ε[i]
        εl = ε[i-1]
        σc = H * εc + squeeze(ae(reshape([εl;σl], (1,2)), [20,20,20,1]))
        taσ = write(taσ, i, σc)
        i+1, taσ
    end
    i = constant(2, dtype=Int32)
    taσ = TensorArray(N)
    taσ = write(taσ, 1, σ0)
    _, out = while_loop(condition, body, [i, taσ])
    set_shape(stack(out), (N,))
end

function visualize(iter)
    full = run(sess, σpred_full)

    close("all")
    plot(d[:,1], d[:,2])
    plot(d[1:end,1], full, "--")
    xlabel("Strain")
    ylabel("Stress (MPa)")
    savefig("figures/strainstress$kid$(tid)_$iter.png")

    close("all")
    ref = d[1:end,2]
    plot(1:n-1,ref[1:n-1], "C1", label="Reference")
    plot(1:n-1,full[1:n-1], "C2--", label="Training Prediction")
    plot(n:size(d,1),ref[n:end], "C1")
    plot(n:size(d,1),full[n:end], "C3--", label="Testing Prediction")
    xlabel("Index")
    ylabel("Stress (MPa)")
    legend()

    savefig("figures/stress$kid$(tid)_$iter.png")
    writedlm("figures/res$kid$(tid)_$iter.txt", full)
end

σpred_train = prediction(d[1:n], σ[1])
loss = sum((σpred_train - σ)^2)

σpred_full = prediction(d[1:end,1], σ[1])

sess = Session(); init(sess)

for i = 1:20
    visualize(i-1)
    BFGS!(sess, loss, 100)
end
