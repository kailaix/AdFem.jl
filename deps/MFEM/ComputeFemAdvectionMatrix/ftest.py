from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np 

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8, "left")
U = FunctionSpace(mesh, "CG", 1)

# Define variational problem
u = TrialFunction(U)
v = TestFunction(U)
a = dot(2.0 * u.dx(0) + 3.0 * u.dx(1), v) * dx

A = assemble(a).array()
DofToVert = vertex_to_dof_map(u.function_space())
A = A[DofToVert,:][:, DofToVert]
print(A.shape)
np.savetxt("fenics/A.txt", A)
