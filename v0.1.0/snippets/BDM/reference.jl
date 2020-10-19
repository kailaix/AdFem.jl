
X = [        0.659027622374092  0.231933368553031
0.659027622374092  0.109039009072877
0.231933368553031  0.659027622374092
0.231933368553031  0.109039009072877
0.109039009072877  0.659027622374092
0.109039009072877  0.231933368553031]
w = [ 0.16666666666666666667
0.16666666666666666667
0.16666666666666666667
0.16666666666666666667
0.16666666666666666667
0.16666666666666666667  ]


function integrate(f)
    s = 0.0
    for i = 1:length(w)
        s += f(X[i,1], X[i,2]) * w[i]
    end
    0.5 * s
end


function calc_basis(ξ, η, x1=1.0, y1=0.0, x2=0.0, y2=1.0, x3=0.0, y3=0.0)
    λ = [ξ;η;1-ξ-η]
    # λ = [1-ξ-η;ξ;η]
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
    return -λ[2]*∇λ[1,:]
end

function φ3(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[2]*∇λ[3,:]
end

function φ4(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return -λ[3]*∇λ[2,:]
end

function φ5(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return λ[3]*∇λ[1,:]
end

function φ6(ξ, η)
    λ, ∇λ = calc_basis(ξ, η)
    return -λ[1]*∇λ[3,:]
end

Basis = [φ1, φ2, φ3, φ4, φ5, φ6]

S = zeros(6,6)
for i = 1:6
    for j = 1:6
        p = (x,y)->Basis[i](x,y)' * Basis[j](x,y)
        S[i,j] = integrate(p)
    end
end
S[ abs.(S) .< 1e-5 ] .= 0.0
@info sum(S), sum(S.^2), sum(S.^3)