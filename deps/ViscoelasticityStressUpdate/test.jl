using Revise
using AdFem
using LinearAlgebra

σ1 = rand(10, 3)
σ2 = zeros(10, 3)
ε1 = rand(10, 3)
ε2 = rand(10, 3)
μ = rand(10)
η = rand(10)
λ = rand(10)
Δt = rand()
for i = 1:10
    s = μ[i]*η[i]*Δt
    S = [
        1 + 2/3*s -1/3*s 0
        -1/3*s 1+2/3*s 0 
        0 0 1+s
    ]
    S = inv(S)
    H = S * [
        2*μ[i]+λ[i] λ[i] 0
        λ[i] 2*μ[i]+λ[i] 0
        0 0 μ[i]
    ]
    σ2[i,:] = H * (ε2[i,:] - ε1[i,:]) + S*σ1[i,:]
end
σ22 = update_stress_viscosity(ε2, ε1, σ1, η, μ, λ, Δt)

@show norm(σ2 - σ22)