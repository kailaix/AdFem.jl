from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np 

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8, "left")
U = VectorFunctionSpace(mesh, "CG", 1, dim=2)

def epsilon(u):
    return sym(nabla_grad(u))

# Define variational problem
u = TrialFunction(U)
v = TestFunction(U)
a = inner(epsilon(u), epsilon(v)) * dx

A = assemble(a).array()
DofToVert = vertex_to_dof_map(u.function_space())
A = A[DofToVert,:][:, DofToVert]
print(A.shape)
np.savetxt("fenics/A.txt", A)
