using SymPy

x, y, t = @syms x y t 
# u1 = exp(-t)*(0.5-x)^2*(2-y)^2
# u2 = exp(-t)*(0.5-x^2)^2*(2-sin(y))^2

u1 = exp(-t)*(2-x)*x*(1-y)
u2 = exp(-t)*(2-x)*x*(1-y)
u11 = diff(u1, x)
u12 = diff(u1, y)
u21 = diff(u2, x)
u22 = diff(u2, y)

ε = [u11; u22; u12+u21]

E = 1.0
ν = 0.35
H = E/(1+ν)/(1-2ν)*[
  1-ν ν 0
  ν 1-ν 0
  0 0 (1-2ν)/2
]

σ = H * ε

σ = [σ[1] σ[3];σ[3] σ[2]]

f1 = u1 - (diff(σ[1,1], x) + diff(σ[1,2], y))
f2 = u2 - (diff(σ[2,1], x) + diff(σ[2,2], y))

s1 = sympy.julia_code(f1)
s1 = replace(replace(s1, ".^"=>"^"), ".*"=>"*")



s2 = sympy.julia_code(f2)
s2 = replace(replace(s2, ".^"=>"^"), ".*"=>"*")


println(s1)

println(s2)