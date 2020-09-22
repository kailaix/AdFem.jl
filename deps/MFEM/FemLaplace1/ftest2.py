from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np 

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8, "left")
U = FunctionSpace(mesh, "CG", 2)

# Define variational problem
u = TrialFunction(U)
v = TestFunction(U)
a = inner(grad(u), grad(v)) * dx

A = assemble(a).array()
dofmap = U.dofmap()
node_dofs = dofmap.dofs(mesh, 0)
edge_dofs = dofmap.dofs(mesh, 1)
dofs = node_dofs + edge_dofs

A = A[dofs,:][:, dofs]
print(A.shape)
np.savetxt("fenics/A2.txt", A)

edges_ = []
for e in edges(mesh):
    vid = [v.index() for v in vertices(e)]
    edges_.append(vid)
edges_ = np.array(edges_, dtype=np.int) + 1
np.savetxt("fenics/edges.txt", edges_, fmt = "%d")