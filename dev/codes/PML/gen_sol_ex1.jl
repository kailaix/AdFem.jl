using SymPy 

x,y,t = @vars x y t 
u = (1-x)*x*(1-y)*y*exp(-t)

β = 1+x^2+y^2 + (1-x)^2+(1-y)^2 

ut = diff(u, t)
utt = diff(ut, t)
f = utt + 2*β*ut + β^2*u
ff = replace(replace(replace(sympy.julia_code(simplify(f)), ".*"=>"*"), ".^"=>"^"), "./"=>"/")
println(ff)