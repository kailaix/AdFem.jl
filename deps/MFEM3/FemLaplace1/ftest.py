from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
# from vedo.dolfin import plot as vplot

# Create mesh and define function space
mesh = UnitCubeMesh(20,20,20)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or \
        x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or \
        x[2] < DOLFIN_EPS or x[2] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("1.0", degree=2)
kappa = Expression("1/(0.1 + x[0]*x[0] + x[1]*x[1] + x[2]*x[2])", degree=2)
a = inner(kappa*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)


##########################################
values = []
for k in range(21):
    for j in range(21):
        for i in range(21):
            values.append(u(0.05*i, 0.05*j, 0.05*k))
values = np.array(values)
np.savetxt("val_all.txt", values)


values = []
for k in range(21):
    values.append(u(0.05*k, 0.5, 0.5))
np.savetxt("val.txt", values)

plt.close("all")
plt.plot(values)
plt.savefig("section_plot.png")

# # Plot solution
# plot(u)
# plt.savefig("fig.png")