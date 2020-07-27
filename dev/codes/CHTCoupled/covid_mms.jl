using SymPy

x, y, t = @vars x y t
T = x * (1-x) * y * (1-y) 
# K = 1 + x^2 + x / (1+y^2)
K = 1.0
u = x * (1-x)* y * (1-y)
v = x * (1-x) * y * (1-y)
p = x * (1-x) * y * (1-y) 
κ1 = 1.0
κ2 = 1.0
w1 = exp(-t) * x * (1-x) * y * (1-y)
w2 = exp(-t) * x * (1-x) * y * (1-y)

Tx = diff(T, x)
Ty = diff(T, y)
k_laplace_T = diff(K * Tx, x) + diff(K * Ty, y)
adv = u * Tx + v * Ty     
Q = adv - k_laplace_T
s = sympy.julia_code(simplify(Q))
s = replace(replace(replace(s, ".*"=>"*"), ".^"=>"^"), "./"=>"/")

ffunc = u * diff(u, x) + v * diff(u, y) + diff(p, x) - 0.01 * (diff(diff(u, x), x) + diff(diff(u, y), y))
gfunc = u * diff(v, x) + v * diff(v, y) + diff(p, y) - 0.01 * (diff(diff(v, x), x) + diff(diff(v, y), y))
f = replace(replace(replace(sympy.julia_code(ffunc), ".*"=>"*"), ".^"=>"^"), "./"=>"/")
g = replace(replace(replace(sympy.julia_code(gfunc), ".*"=>"*"), ".^"=>"^"), "./"=>"/")

div_term = diff(u, x) + diff(v, y)
div_term = replace(replace(replace(sympy.julia_code(div_term), ".*"=>"*"), ".^"=>"^"), "./"=>"/")


w1t = diff(w1, t)
w2t = diff(w2, t)
q1_func = w1t  - κ1 * (u-w1)
q1 = replace(replace(replace(sympy.julia_code(q1_func), ".*"=>"*"), ".^"=>"^"), "./"=>"/")
q2_func = w2t  - κ2 * (v-w2)
q2 = replace(replace(replace(sympy.julia_code(q2_func), ".*"=>"*"), ".^"=>"^"), "./"=>"/")
