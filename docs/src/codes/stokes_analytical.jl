# div σ = f
# div u = g
# σ = -p I + 2η ε(u)

using SymPy 

x, y = @vars x y 
u = x*(1-x)*y*(1-y)
v = x^2*(1-x^2)*y^2*(1-y^2)
p = x + y 

g = diff(u, x) + diff(v, y)

ε = [diff(u,x) 1/2*(diff(u,y) + diff(v, x))
1/2*(diff(u,y) + diff(v, x)) diff(v,y)]
σ = -[p 0;0 p] + 2 * ε

f1 = diff(σ[1,1], x) + diff(σ[1,2], y)
f2 = diff(σ[2,1], x) + diff(σ[2,2], y)

print(replace(replace(sympy.julia_code(g), ".*"=>"*"), ".^"=>"^"))

print(replace(replace(sympy.julia_code(f1), ".*"=>"*"), ".^"=>"^"))
print(replace(replace(sympy.julia_code(f2), ".*"=>"*"), ".^"=>"^"))