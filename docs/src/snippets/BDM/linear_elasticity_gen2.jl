using SymPy
using LinearAlgebra
λ = μ = 1.0

function gen_code(u)
    s = replace(replace(replace(sympy.julia_code(simplify(u)), ".*"=>"*"), ".^"=>"^"), "./"=>"/")
    "  (x,y)->begin;$s;end,\n"
end

function gen_code(u, v, g, h)
    s1 = gen_code(u)
    s2 = gen_code(v)
    s3 = gen_code(g)
    s4 = gen_code(h)
    "(\n" * s1 * s2 * s3 * s4 * ")"
end

x, y = @vars x y
# u = x*(1-x)*y*(1-y)
# v = x*(1-x)*y*(1-y)


# u = x^2 * (1-x) * y^2 * (1-y^2)
# v = u 

u = x^2 * (1-x) * y^2 * (1-y^2)
v = x*(1-x)*y*(1-y)


ux = diff(u, x)
uy = diff(u, y)
vx = diff(v, x)
vy = diff(v, y)

ε = [
    ux (uy+vx)/2
    (uy+vx)/2 vy 
]

σ = 2μ * ε + λ*(ux + vy) * diagm(0=>ones(2))

g = diff(σ[1,1], x) + diff(σ[1,2], y)
h = diff(σ[2,1], x) + diff(σ[2,2], y)

println(gen_code(u, v, g, h))
