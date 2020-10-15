using PyPlot 
using LinearAlgebra
g1 = 1/2-√3/6
g2 = 1/2+√3/6

function calc_basis(ξ, η, x1=1.0, y1=0.0, x2=0.0, y2=1.0, x3=0.0, y3=0.0)
    λ = [ξ;η;1-ξ-η]
    K = 0.5 * abs(det([
        x2-x1 x3-x1 
        y2-y1 y3-y1
    ]))
    ∇λ = [
        y2-y3 x3-x2 
        y3-y1 x1-x3 
        y1-y2 x2-x1 
    ]/2K 
    ∇λ = [∇λ[:,2] -∇λ[:,1]]
    return λ, ∇λ
end


function φ1(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[1]*∇λ[2,:]
end

function φ2(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[2]*∇λ[1,:]
end

function φ3(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[2]*∇λ[3,:]
end

function φ4(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[3]*∇λ[2,:]
end

function φ5(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[3]*∇λ[1,:]
end

function φ6(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[1]*∇λ[3,:]
end

Basis = [φ1, φ2, φ3, φ4, φ5, φ6]

a = LinRange(0, 1, 10)
ξ = zeros(length(a)^2)
η = zeros(length(a)^2)
k = 0
for i = 1:length(a)
    for j = 1:length(a)
        global k = k + 1
        ξ[k] = a[i]
        η[k] = a[j]
    end
end
Idx = ξ + η .<= 1
ξ = ξ[Idx]
η = η[Idx]

m = Dict(1=>1, 2=>4, 3=>2, 4=>5, 5=>3, 6=>6)
figure(figsize=(15, 10))
for k = 1:6
    subplot(230+m[k])
    U = zeros(length(ξ))
    V = zeros(length(ξ))
    for i = 1:length(ξ)
        U[i], V[i] = Basis[k](ξ[i], η[i])
    end
    plot([0;1;0;0], [0;0;1;0])
    quiver(ξ, η, U, V)
    title("\$\\phi_$k\$")
    axis("equal")
    xlim(-0.25,1.25)
    ylim(-0.25,1.25)
    if m[k] in [4,5,6]
        xlabel("\$\\xi\$")
    end
    if m[k] in [1,4]
        ylabel("\$\\eta\$")
    end
end
savefig("BDM.png")