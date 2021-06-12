We use the following parameters for simulation

| Parameter | Value     |
|-----------|-----------|
| a         | 0.0125    |
| b         | 0.0172    |
| $d_C$     | 0.037     |
| $V_0$     | $10^{-6}$ |
| $f_0$     | 0.6       |


```julia
using AdFem
using PyPlot

T = 1e-10
NT = 100
Δt = T/NT 
dc = 0.037
v0 = 1e-6
f0 = 0.6
a = 0.0125
b = 0.0172


ψ = TensorArray(NT+1)
ψ = write(ψ, 1, zeros(1))
V = ones(NT+1) * v0 
# V[NT÷5] = 0.0001
V = reshape(V, :, 1)
V = constant(V)

function condition(i, ψ)
    i <= NT+1
end
function body(i, ψ)
    ψ0 = read(ψ, i-1)
    v = V[i]
    ψ1 = solve_slip_law(v, ψ0, dc, v0, a, b, f0, Δt::Float64)
    i+1, write(ψ, i, ψ1)
end

i = constant(2, dtype = Int32)
_, out = while_loop(condition, body, [i, ψ])
out = stack(out)

sess = Session(); init(sess)
Ψ = run(sess, out)
close("all")
plot(Ψ[:])
savefig("test.png")

```