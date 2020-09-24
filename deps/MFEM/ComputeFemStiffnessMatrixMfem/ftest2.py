from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np 


# Create mesh and define function space
mesh = UnitSquareMesh(8, 8, "left")
U = VectorFunctionSpace(mesh, "CG", 2, dim = 2)

def epsilon(u):
    return sym(nabla_grad(u))

# Define variational problem
u = TrialFunction(U)
v = TestFunction(U)
a = inner(epsilon(u), epsilon(v)) * dx


A = assemble(a).array()
dofmap = U.dofmap()
node_dofs = dofmap.dofs(mesh, 0)
edge_dofs = dofmap.dofs(mesh, 1)
dofs = []
nv = int(len(node_dofs)/2)
ne = int(len(edge_dofs)/2)
for i in range(nv):
    dofs.append(node_dofs[2*i])
for i in range(ne):
    dofs.append(edge_dofs[2*i])
for i in range(nv):
    dofs.append(node_dofs[2*i+1])
for i in range(ne):
    dofs.append(edge_dofs[2*i+1])
    
A = A[dofs,:][:, dofs]
print(A.shape)
np.savetxt("fenics/A2.txt", A)

edges_ = []
for e in edges(mesh):
    vid = [v.index() for v in vertices(e)]
    edges_.append(vid)
edges_ = np.array(edges_, dtype=np.int) + 1
np.savetxt("fenics/edges.txt", edges_, fmt = "%d")