# Conjugate Heat Transfer (CHT) Energy equation

The steady-state energy equation for the conjugate heat transfer problem in two spatial dimensions is given by

$$\rho\,  C_p \, {\bf u}\cdot \nabla T  = \nabla \cdot (k \nabla T) + Q,$$

i.e., 

$$\rho\,  C_p\, \left(u \frac{\partial T}{\partial x}+v \frac{\partial T}{\partial y}\right)=k\left(\frac{\partial^{2} T}{\partial x^{2}}+\frac{\partial^{2} T}{\partial y^{2}}\right)+Q,$$

where $c$ is the specific heat capacity, $\rho$ is the density, ${\bf u} = [u\  v]$ is the steady-state velocity field, $T$ is the steady-state tempearture, $k$ is the conductivity, and $Q$ is the heat source.

The transient energy equation for the conjugate heat transfer problem in two spatial dimensions is given by

$$\rho\,  C_p \, \frac{\partial T}{\partial t}  + \rho\,  C_p \, {\bf u}\cdot \nabla T  = \nabla \cdot (k \nabla T) + Q,$$

