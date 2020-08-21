# Steady-state Navier-Stokes equations

The Navier-Stokes equations describe the motion of viscous flow formed by a fluid material. Assume the material has density $\rho$, dynamic viscosity $\mu$, velocity $\bf u$, pressure $p$, and body accelerations $\bf g$, the Navier-Stokes equations include the the continuity equation, which describes the conservation of mass:

$$\frac{\partial \rho}{\partial t} + \sum_i \frac{\partial \rho u_i}{\partial x_i} = 0,$$

and the momentum equation, which describes the conservation of momentum in each spatial direction:

$$\frac{\partial \rho u_i}{\partial t} + \sum_j \frac{\partial \rho u_i u_j}{\partial x_j} = \sum_j \frac{\partial \tau_{ij}}{\partial x_j} - \frac{\partial p }{\partial x_i}+\rho g_i, \quad \forall i,$$

where the stress tensor $\tau_{ij}$ is defined as
$$\tau_{ij} = \mu \left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) - \frac{2}{3} \mu \delta_{ij} \frac{\partial u_j}{\partial x_j}.$$

# Steady-state Navier-Stokes equations for incompressible flow

We assume the fluid material is incompressible with constant density $\rho$, and we denote its the kinematic viscosity as $\nu=\mu / \rho$.
We assume that the system has reached a steady state. Then, the steady-state incompressible Navier-Stokes equations in two spatial dimensions are given by

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0 \tag{1}$$
$$u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}\right)+f \tag{2}$$
$$u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}}\right)+g \tag{3}$$

where (1) is the continuity equation and (2)-(3) are the momentum equations.

## The Newton's method

Let $\delta u'$ denote the finite element basis for $u$, and $\delta v'$ denote the finite element basis for $v$. To derive the weak form, we multiply both sides of (2)-(3) by $\delta u'$ and $\delta v'$, respectively.

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) =  -\frac{1}{\rho} \left(\frac{\partial p}{\partial x}, \delta u'\right)+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}, \ \delta u'\right)+\left(f, \delta u'\right)$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) = -\frac{1}{\rho}\left( \frac{\partial p}{\partial y}, \delta v'\right) +\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}} ,\ \delta v'\right)+ \left( g, \delta v'\right)$$



Then we have the following weak form

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) =  \frac{1}{\rho} \left(p, \ \frac{\partial \delta u'}{\partial x}\right)-\nu\left(\nabla u, \nabla \delta u'\right)+\left(f, \delta u'\right) \tag{4}$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) = \frac{1}{\rho}\left(p, \frac{\partial \delta v'}{\partial y}\right) -\nu\left(\nabla v, \nabla\delta v'\right)+ \left( g, \delta v'\right) \tag{5}$$

Additionally, we multiply both sides of (1) by $\delta p'$, then we have

$$\left(\frac{\partial u}{\partial x}, \delta p'\right) + \left(\frac{\partial v}{\partial y}, \delta p' \right) = 0 \tag{6}$$

The weak form (4),(5), and (6) are nonlinear in $u$ and $v$. We use the Newton's method to solve coupled system  iteratively.

To this end, we define the residual functions

$$F(u,v) = \left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) -  \frac{1}{\rho} \left(p, \ \frac{\partial \delta u'}{\partial x}\right)+ \nu\left(\nabla u, \nabla \delta u'\right)-\left(f, \delta u'\right)$$

$$G(u,v) = \left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) -\frac{1}{\rho}\left(p, \frac{\partial \delta v'}{\partial y}\right) +\nu\left(\nabla v, \nabla\delta v'\right)- \left( g, \delta v'\right)$$

$$H(u, v) = \left(\frac{\partial u}{\partial x}, \delta p'\right) + \left(\frac{\partial v}{\partial y}, \delta p' \right)$$

we have the following equation for one iteration of the Newton's method

$$\begin{bmatrix}\nabla_u F(u,v)  & \nabla_v F(u,v) & \nabla_p F(u, v)  \\ \nabla_u G(u,v)  & \nabla_v G(u,v) & \nabla_p G(u, v) \\ \nabla_u H(u, v) & \nabla_v H(u,v) & 0\end{bmatrix} \begin{bmatrix}\Delta u\\ \Delta v \\ \Delta p\end{bmatrix} = - \begin{bmatrix}F(u,v) \\ G(u,v)\\H(u,v) \end{bmatrix} \tag{4}$$

$$\begin{bmatrix} u_{new}\\ v_{new} \\ p_{new} \end{bmatrix} = \begin{bmatrix} u\\ v \\ p\end{bmatrix} + \begin{bmatrix}\Delta u\\ \Delta v \\\Delta p \end{bmatrix}$$



We use Taylor's expansion to linearize $F(u+\Delta u, v+\Delta v), G(u+\Delta u, v+\Delta v)$ and obtain

$$F(u+\Delta u, v+\Delta v) = F(u,v) + \nabla_u F(u,v)\Delta u   + \nabla_v F(u,v) \Delta v$$

$$G(u+\Delta u, v+\Delta v) = G(u,v) + \nabla_u G(u,v) \Delta u + \nabla_v G(u,v)\Delta v$$

Thus, we have

$$\nabla_u F(u,v)\Delta u = \left(\Delta u \frac{\partial u}{\partial x}, \delta u'\right) +\left(u \frac{\partial \Delta u }{\partial x}, \delta u'\right)+ \left(v \frac{\partial \Delta u }{\partial y}, \delta u'\right) + (\nu\nabla (\Delta u), \nabla \delta u')$$

$$\nabla_v F(u,v)\Delta v = \left(\Delta v \frac{\partial u}{\partial y}, \delta u'\right)$$

$$\nabla_u G(u,v)\Delta u = \left(\Delta u \frac{\partial v}{\partial x}, \delta v'\right)$$

$$\nabla_v G(u,v)\Delta v = \left(\Delta v \frac{\partial v}{\partial y}, \delta v'\right) +\left(v \frac{\partial \Delta v }{\partial y}, \delta v'\right)+ \left(u \frac{\partial \Delta v }{\partial x}, \delta v'\right) +ν (\nabla (\Delta v), \nabla \delta v')$$

# Numerical experiments

Results of numerical experiments are summarized in the table below. In each case, we assume that the steady-state velocity field is measured at a sample of locations, represented by a sample of grid points in the finite element method. We solve the optimization problem to minimize the difference between the measured velocity and the computed velocity based on the current model parameters. In order to model the uncertainty in the measured data, we include a noise level when we transform the solution to the forwrad problem into measured data: each data will be multiplied by a random scalar which is uniform on (1-noise_level, 1+noise_level)


| grid size | sample size | noise level | number of iterations | exact viscosity |  predicted viscosity |
|:---------:|:-----------:|:-----------:|:--------------:|:-----------------:|:--------------------:|
|  20 by 20 |     20     |      0      |       28       |        0.01       | 0.010000000000000004 |
|  20 by 20 |     20     |    0.01     |       26       |        0.01       | 0.010025030255978819 |
|  20 by 20 |     20     |    0.05     |       27       |        0.01       | 0.009872678483953757 |
|  20 by 20 |     20     |    0.10     |       28       |        0.01       | 0.009981956568710024 |
|  20 by 20 |     20     |    0.20     |       28       |        0.01       | 0.008823737264381126 |
|  20 by 20 |     20     |    0.20     |       29       |        0.01       | 0.010599176390172771 |
|  20 by 20 |     20     |    0.20     |       50       |        0.01       | 0.009202237809712532 |
