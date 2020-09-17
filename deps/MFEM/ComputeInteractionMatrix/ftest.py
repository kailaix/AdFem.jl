
from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np 

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8, "left")
plot(mesh)
plt.gca().invert_yaxis()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("fenics/mesh.png")
P = FunctionSpace(mesh, 'DG', 0)
U = FunctionSpace(mesh, "CG", 1)

# Define variational problem
u = TrialFunction(U)
p = TestFunction(P)

a = dot(p , (u.dx(0) + u.dx(1))) * dx 

A = assemble(a).array()
print(A.shape)
DofToVert = vertex_to_dof_map(u.function_space())

np.savetxt("fenics/A.txt", A[:, DofToVert])
