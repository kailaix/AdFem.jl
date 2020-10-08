using PyPlot 

g1 = 1/2-√3/6
g2 = 1/2+√3/6
function φ1(ξ, η)
    return √2/(g2-g1) * [g2*ξ; (g2-1)*η]
end

function φ2(ξ, η)
    return √2/(g1-g2) * [g1*ξ; (g1-1)*η]
end

function φ3(ξ, η)
    return 1/(g2-g1)*[g2*ξ+η-g2; (g2-1)*η]
end

function φ4(ξ, η)
    return 1/(g1-g2)*[g1*ξ+η-g1; (g1-1)*η]
end

function φ5(ξ, η)
    return 1/(g2-g1)*[(g2-1)*ξ; ξ+g2*η-g2]
end

function φ6(ξ, η)
    return 1/(g1-g2)*[(g1-1)*ξ; ξ+g1*η-g1]
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