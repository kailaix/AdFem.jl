from fenics import *
import matplotlib.pyplot as plt 

V = VectorFunctionSpace ( mesh, "CG", 1 )
u = Function ( V )
v = TestFunction ( V )

F = inner(grad(u)*u, v)*dx + nu*inner(grad(u), grad(v))*dx 
J = derivative ( F, u )

solve ( F == 0, u, bc, J = J )
plot ( u, title = 'burgers steady viscous equation' )
plt.savefig ( 'burgers_steady_viscous.png' )
