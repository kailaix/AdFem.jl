# div σ = f
# div u = g
# σ = -p I + 2η ε(u)

using SymPy 

x, y = @vars x y 
u = 2*pi*sin(pi*x)*sin(pi*x)*cos(pi*y)*sin(pi*y)
v = -2*pi*sin(pi*x)*sin(pi*y)*cos(pi*x)*sin(pi*y)
p = sin(pi*x)*sin(pi*y)
nu = 0.5

f1 = diff(p, x) - nu * (diff(diff(u, x), x) + diff(diff(u, y), y) )
f2 = diff(p, y) - nu * (diff(diff(v, x), x) + diff(diff(v, y), y) )

println(replace(replace(sympy.julia_code(f1), ".*"=>"*"), ".^"=>"^"))
println(replace(replace(sympy.julia_code(f2), ".*"=>"*"), ".^"=>"^"))