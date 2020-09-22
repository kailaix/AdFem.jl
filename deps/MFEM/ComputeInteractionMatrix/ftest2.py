
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
U = FunctionSpace(mesh, "CG", 2)

# Define variational problem
u = TrialFunction(U)
p = TestFunction(P)

a = dot(p , u.dx(0)) * dx 
b = dot(p , u.dx(1)) * dx 

dofmap = U.dofmap()
node_dofs = dofmap.dofs(mesh, 0)
edge_dofs = dofmap.dofs(mesh, 1)
dofs = node_dofs + edge_dofs

A = assemble(a).array()
np.savetxt("fenics/A2.txt", A[:, dofs])

A = assemble(b).array()
print(A.shape)
np.savetxt("fenics/B2.txt", A[:, dofs])

edges_ = []
for e in edges(mesh):
    vid = [v.index() for v in vertices(e)]
    edges_.append(vid)
edges_ = np.array(edges_, dtype=np.int) + 1
np.savetxt("fenics/edges.txt", edges_, fmt = "%d")
