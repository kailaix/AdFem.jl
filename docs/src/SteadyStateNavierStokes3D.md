# Steady-state Navier-Stokes equations in 3D space

The Navier-Stokes equations describe the motion of viscous flow formed by a fluid material. Assume the material has density $\rho$, dynamic viscosity $\mu$, velocity $\bf u$, pressure $p$, and body accelerations $\bf g$, the Navier-Stokes equations include the the continuity equation, which describes the conservation of mass:

$$\frac{\partial \rho}{\partial t} + \sum_i \frac{\partial \rho u_i}{\partial x_i} = 0,$$

and the momentum equation, which describes the conservation of momentum in each spatial direction:

$$\frac{\partial \rho u_i}{\partial t} + \sum_j \frac{\partial \rho u_i u_j}{\partial x_j} = \sum_j \frac{\partial \tau_{ij}}{\partial x_j} - \frac{\partial p }{\partial x_i}+\rho g_i, \quad \forall i,$$

where the stress tensor $\tau_{ij}$ is defined as
$$\tau_{ij} = \mu \left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) - \frac{2}{3} \mu \delta_{ij} \frac{\partial u_j}{\partial x_j}.$$

# Steady-state Navier-Stokes equations for incompressible flow

We assume the fluid material is incompressible with constant density $\rho$, and we denote its the kinematic viscosity as $\nu=\mu / \rho$.
We assume that the system has reached a steady state. Then, the steady-state incompressible Navier-Stokes equations in three spatial dimensions are given by

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}+\frac{\partial w}{\partial z}=0 \tag{1}$$
$$u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}+w \frac{\partial u}{\partial z}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}+\frac{\partial^{2} u}{\partial z^{2}}\right)+f \tag{2}$$
$$u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}+w \frac{\partial v}{\partial z}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}}+\frac{\partial^{2} v}{\partial z^{2}}\right)+g \tag{3}$$
$$u \frac{\partial w}{\partial x}+v \frac{\partial w}{\partial y}+w \frac{\partial w}{\partial z}=-\frac{1}{\rho} \frac{\partial p}{\partial z}+\nu\left(\frac{\partial^{2} w}{\partial x^{2}}+\frac{\partial^{2} w}{\partial y^{2}}+\frac{\partial^{2} w}{\partial z^{2}}\right)+h \tag{4}$$

where (1) is the continuity equation and (2)-(4) are the momentum equations.

## The Newton's method

Let $\delta u'$ denote the finite element basis for $u$, $\delta v'$ denote the finite element basis for $v$, and $\delta w'$ denote the finite element basis for $w$. To derive the weak form, we multiply both sides of (2)-(4) by $\delta u'$, $\delta v'$, and $\delta w'$, respectively.

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right)+ \left(w \frac{\partial u}{\partial z} , \delta u'\right) =  -\frac{1}{\rho} \left(\frac{\partial p}{\partial x}, \delta u'\right)+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}+\frac{\partial^{2} u}{\partial z^{2}}, \ \delta u'\right)+\left(f, \delta u'\right)$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right)+ \left(w \frac{\partial v}{\partial z}, \delta v'\right) = -\frac{1}{\rho}\left( \frac{\partial p}{\partial y}, \delta v'\right) +\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}}+\frac{\partial^{2} v}{\partial z^{2}} ,\ \delta v'\right)+ \left( g, \delta v'\right)$$

$$\left(u \frac{\partial w}{\partial x}, \delta w'\right) + \left(v \frac{\partial w}{\partial y}, \delta w'\right)+ \left(w \frac{\partial w}{\partial z}, \delta w'\right) = -\frac{1}{\rho}\left( \frac{\partial p}{\partial z}, \delta w'\right) +\nu\left(\frac{\partial^{2} w}{\partial x^{2}}+\frac{\partial^{2} w}{\partial y^{2}}+\frac{\partial^{2} w}{\partial z^{2}} ,\ \delta w'\right)+ \left( h, \delta w'\right)$$

Then we have the following weak form

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) + \left(w \frac{\partial u}{\partial z} , \delta u'\right) =  \frac{1}{\rho} \left(p, \ \frac{\partial \delta u'}{\partial x}\right)-\nu\left(\nabla u, \nabla \delta u'\right)+\left(f, \delta u'\right) \tag{5}$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) + \left(w \frac{\partial v}{\partial z}, \delta v'\right) = \frac{1}{\rho}\left(p, \frac{\partial \delta v'}{\partial y}\right) -\nu\left(\nabla v, \nabla\delta v'\right)+ \left( g, \delta v'\right) \tag{6}$$

$$\left(u \frac{\partial w}{\partial x}, \delta w'\right) + \left(v \frac{\partial w}{\partial y}, \delta w'\right) + \left(w \frac{\partial w}{\partial z}, \delta w'\right) = \frac{1}{\rho}\left(p, \frac{\partial \delta w'}{\partial z}\right) -\nu\left(\nabla v, \nabla\delta w'\right)+ \left( h, \delta w'\right) \tag{7}$$

Additionally, we multiply both sides of (1) by $\delta p'$, then we have

$$\left(\frac{\partial u}{\partial x}, \delta p'\right) + \left(\frac{\partial v}{\partial y}, \delta p' \right)+ \left(\frac{\partial w}{\partial z}, \delta p' \right) = 0 \tag{8}$$

The weak form (4),(5), and (6) are nonlinear in $u$, $v$ and $w$. We use the Newton's method to solve coupled system  iteratively.

To this end, we define the residual functions

$$F(u,v,w) = \left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) + \left(w \frac{\partial u}{\partial z} , \delta u'\right) -  \frac{1}{\rho} \left(p, \ \frac{\partial \delta u'}{\partial x}\right)+ \nu\left(\nabla u, \nabla \delta u'\right)-\left(f, \delta u'\right)$$

$$G(u,v,w) = \left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right)+ \left(w \frac{\partial v}{\partial z}, \delta v'\right) -\frac{1}{\rho}\left(p, \frac{\partial \delta v'}{\partial y}\right) +\nu\left(\nabla v, \nabla\delta v'\right)- \left( g, \delta v'\right)$$

$$H(u,v,w) = \left(u \frac{\partial w}{\partial x}, \delta w'\right) + \left(v \frac{\partial w}{\partial y}, \delta w'\right)+ \left(w \frac{\partial w}{\partial z}, \delta w'\right) -\frac{1}{\rho}\left(p, \frac{\partial \delta w'}{\partial z}\right) +\nu\left(\nabla w, \nabla\delta w'\right)- \left( h, \delta w'\right)$$

$$I(u, v,w) = \left(\frac{\partial u}{\partial x}, \delta p'\right) + \left(\frac{\partial v}{\partial y}, \delta p' \right)+\left(\frac{\partial w}{\partial z}, \delta p' \right)$$

we have the following equation for one iteration of the Newton's method

$$\begin{bmatrix}\nabla_u F(u, v, w)  & \nabla_v F(u, v, w) & \nabla_w F(u, v, w) & \nabla_p F(u, v, w)  \\ \nabla_u G(u, v, w)  & \nabla_v G(u, v, w) & \nabla_w G(u, v, w) & \nabla_p G(u, v, w) \\ \nabla_u H(u, v, w) & \nabla_v H(u, v, w) & \nabla_w H(u, v, w)& \nabla_p H(u, v, w)  \\ \nabla_u I(u, v, w) & \nabla_v I(u, v, w) & \nabla_w I(u, v, w)& 0 \end{bmatrix} \begin{bmatrix}\Delta u\\ \Delta v \\ \Delta w \\ \Delta p\end{bmatrix} = - \begin{bmatrix}F(u, v, w) \\ G(u, v, w)\\H(u, v, w) \\I(u, v, w) \end{bmatrix} \tag{9}$$

$$\begin{bmatrix} u_{new}\\ v_{new}\\ w_{new} \\ p_{new} \end{bmatrix} = \begin{bmatrix} u\\ v \\ w \\ p\end{bmatrix} + \begin{bmatrix}\Delta u\\ \Delta v  \\ \Delta w \\\Delta p \end{bmatrix}$$



We use Taylor's expansion to linearize $F(u+\Delta u, v+\Delta v, w+\Delta w), G(u+\Delta u, v+\Delta v, w+\Delta w), H(u+\Delta u, v+\Delta v, w+\Delta w)$ and obtain

$$F(u+\Delta u, v+\Delta v, w+\Delta w) = F(u, v, w) + \nabla_u F(u, v, w)\Delta u   + \nabla_v F(u, v, w) \Delta v+ \nabla_w F(u, v, w) \Delta w$$

$$G(u+\Delta u, v+\Delta v, w+\Delta w) = G(u, v, w) + \nabla_u G(u, v, w)\Delta u   + \nabla_v G(u, v, w) \Delta v+ \nabla_w G(u, v, w) \Delta w$$

$$H(u+\Delta u, v+\Delta v, w+\Delta w) = H(u, v, w) + \nabla_u H(u, v, w)\Delta u   + \nabla_v H(u, v, w) \Delta v+ \nabla_w H(u, v, w) \Delta w$$

Thus, we have

$$\nabla_u F(u, v, w)\Delta u = \left(\Delta u \frac{\partial u}{\partial x}, \delta u'\right) +\left(u \frac{\partial \Delta u }{\partial x}, \delta u'\right)+ \left(v \frac{\partial \Delta u }{\partial y}, \delta u'\right)+ \left(w \frac{\partial \Delta u }{\partial z}, \delta u'\right) + (\nu\nabla (\Delta u), \nabla \delta u')$$

$$\nabla_v F(u, v, w)\Delta v = \left(\Delta v \frac{\partial u}{\partial y}, \delta u'\right)$$

$$\nabla_w F(u, v, w)\Delta w = \left(\Delta w \frac{\partial u}{\partial z}, \delta u'\right)$$

$$\nabla_u G(u, v, w)\Delta u = \left(\Delta u \frac{\partial v}{\partial x}, \delta v'\right)$$

$$\nabla_v G(u, v, w)\Delta v = \left(\Delta v \frac{\partial v}{\partial y}, \delta v'\right)+ \left(u \frac{\partial \Delta v }{\partial x}, \delta v'\right) +\left(v \frac{\partial \Delta v }{\partial y}, \delta v'\right) +\left(w \frac{\partial \Delta v }{\partial z}, \delta v'\right) + (ν\nabla (\Delta v), \nabla \delta v')$$

$$\nabla_w G(u, v, w)\Delta w = \left(\Delta w \frac{\partial v}{\partial z}, \delta v'\right)$$

$$\nabla_u H(u, v, w)\Delta u = \left(\Delta u \frac{\partial w}{\partial x}, \delta w'\right)$$

$$\nabla_v H(u, v, w)\Delta v = \left(\Delta v \frac{\partial w}{\partial y}, \delta w'\right)$$

$$\nabla_w H(u, v, w)\Delta w = \left(\Delta w \frac{\partial w}{\partial z}, \delta w'\right)+ \left(u \frac{\partial \Delta w }{\partial x}, \delta w'\right) +\left(v \frac{\partial \Delta w }{\partial y}, \delta w'\right) +\left(w \frac{\partial \Delta w }{\partial z}, \delta w'\right) + (ν\nabla (\Delta w), \nabla \delta w')$$
