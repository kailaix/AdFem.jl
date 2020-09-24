from __future__ import print_function
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np 

# Create mesh and define function space
mesh = UnitSquareMesh(50, 50, "left")

# V = VectorElement("Lagrange",triangle, 2)
# Q = FiniteElement("Lagrange",triangle, 1)
V = VectorElement("Lagrange",triangle, 2)
Q = FiniteElement("DG",triangle, 0)

element = MixedElement([V, Q])
W = FunctionSpace(mesh,element)

# Define variational problem
up = Function(W)
u, p = split(up)
v, q = TestFunctions(W)

F = inner(grad(u)*u, v)*dx + inner(grad(u), grad(v))*dx \
     - inner(p, div(v))*dx + inner(q, div(u))*dx

noslip  = DirichletBC(W.sub(0), (0, 0), "on_boundary")
inlet = DirichletBC(W.sub(0), Constant((1.0,0.00)), "on_boundary && x[1] > 0.99")
bcu = [noslip,inlet]

solve(F == 0, up, bcu)

def plot_index(k):
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    xs, ys = np.meshgrid(x, y)
    zs = np.zeros(xs.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            zs[i,j] = up(np.array([xs[i, j], ys[i, j]]))[k]

    plt.close("all")
    plt.pcolormesh(xs, ys, zs)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"u{k}.png")

plot_index(0)
plot_index(1)
plot_index(2)