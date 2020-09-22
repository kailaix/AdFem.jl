
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
a = dot(p, u.dx(0))*dx
b = dot(p, u.dx(1))*dx


A = assemble(a).array().T
x = np.random.random((A.shape[1],))
f = np.dot(A, x)
A1 = assemble(b).array().T
f1 = np.dot(A1, x)

dofmap = U.dofmap()
node_dofs = dofmap.dofs(mesh, 0)
edge_dofs = dofmap.dofs(mesh, 1)
dofs = node_dofs + edge_dofs

np.savetxt("fenics/f2.txt", np.concatenate([f[dofs], f1[dofs]]))
np.savetxt("fenics/x2.txt", x)


edges_ = []
for e in edges(mesh):
    vid = [v.index() for v in vertices(e)]
    edges_.append(vid)
edges_ = np.array(edges_, dtype=np.int) + 1
np.savetxt("fenics/edges.txt", edges_, fmt = "%d")