# Conjugate heat transfer coupled solver for incompressible fluid

We couple the Navier-Stokes equations with the energy equation to obtain the following nonlinear system of $u, v, p, T$ in two spatial dimensions:

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0 \tag{1}$$
$$u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\frac{\partial }{\partial x}\left(\nu \frac{\partial  u}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  u}{\partial y } \right)+f_1 \tag{2}$$
$$u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\frac{\partial }{\partial x}\left(\nu \frac{\partial  v}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  v}{\partial y } \right)+f_2 \tag{3}$$
$$\rho\,  C_p \, \left( u \frac{\partial T}{\partial x}+v \frac{\partial T}{\partial y} \right)=\frac{\partial }{\partial x}\left(k \frac{\partial  T}{\partial x } \right)+ \frac{\partial }{\partial y}\left(k \frac{\partial  T}{\partial y } \right)+Q \tag{4}$$
where ${\bf u} = [u\  v]$ is the steady-state velocity field, $\rho$ is the density, $C_p$ is the specific heat capacity,  $T$ is the steady-state tempearture, $k$ is the conductivity, and $Q$ is the heat source.

Let $\delta u'$ denote the finite element basis for $u$, $\delta v'$ denote the finite element basis for $v$, and $\delta T'$ denote the finite element basis for $T$. To derive the weak form of (4)-(5), we multiply both sides of (2)-(4) by $\delta u'$，$\delta v'$,  and $\delta T'$, respectively.

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) =  -\frac{1}{\rho} \left(\frac{\partial p}{\partial x}, \delta u'\right)+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}, \ \delta u'\right)+\left(f_1, \delta u'\right)$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) = -\frac{1}{\rho}\left( \frac{\partial p}{\partial y}, \delta v'\right) +\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}} ,\ \delta v'\right)+ \left( f_2, \delta v'\right)$$

$$\rho \, C_p \left(u \frac{\partial T}{\partial x}, \delta T'\right) + \rho \, C_p \left(v \frac{\partial T}{\partial y}, \delta T'\right) = k \left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}} ,\ \delta T'\right)+ \left( Q, \delta T'\right)$$

If $\nu$ and $k$ are space-varying, we have

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) =  -\frac{1}{\rho} \left(\frac{\partial p}{\partial x}, \delta u'\right)+\left(\frac{\partial }{\partial x}\left(\nu \frac{\partial  u}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  u}{\partial y } \right), \ \delta u'\right)+\left(f_1, \delta u'\right)$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) = -\frac{1}{\rho}\left( \frac{\partial p}{\partial y}, \delta v'\right) +\left(\frac{\partial }{\partial x}\left(\nu \frac{\partial  u}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  u}{\partial y } \right) ,\ \delta v'\right)+ \left( f_2, \delta v'\right)$$

$$\rho \, C_p \left(u \frac{\partial T}{\partial x}, \delta T'\right) + \rho \, C_p \left(v \frac{\partial T}{\partial y}, \delta T'\right) =  \left(\frac{\partial }{\partial x}\left(k \frac{\partial  T}{\partial x } \right)+ \frac{\partial }{\partial y}\left(k \frac{\partial  T}{\partial y } \right) ,\ \delta T'\right)+ \left( Q, \delta T'\right)$$

Then we have the following weak form

$$\left(u \frac{\partial u}{\partial x}, \delta u'\right)+ \left(v \frac{\partial u}{\partial y} , \delta u'\right) =  \frac{1}{\rho} \left(p, \ \frac{\partial \delta u'}{\partial x}\right)-\nu\left(\nabla u, \nabla \delta u'\right)+\left(f_1, \delta u'\right)$$

$$\left(u \frac{\partial v}{\partial x}, \delta v'\right) + \left(v \frac{\partial v}{\partial y}, \delta v'\right) = \frac{1}{\rho}\left(p, \frac{\partial \delta v'}{\partial y}\right) -\nu\left(\nabla v, \nabla\delta v'\right)+ \left( f_2, \delta v'\right)$$

$$\rho \, C_p \left(u \frac{\partial T}{\partial x}, \delta T'\right) + \rho \, C_p \left(v \frac{\partial T}{\partial y}, \delta T'\right) = - k\left(\nabla T, \nabla\delta T'\right)+ \left( Q, \delta T'\right)$$
We use the Newton's method to solve the problem iteratively.

Temperature variations within a convective flow give rise to variations in fluid properties (e.g., density and viscosity). In Boussinesq approximations, all density varitions are neglected. This results in the following system of governing equations:
 
$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0 \tag{1}$$
$$u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}=- \frac{1}{\rho}\frac{\partial p}{\partial x}+\frac{\partial }{\partial x}\left( \nu \frac{\partial  u}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  u}{\partial y } \right)\tag{2}$$
$$u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y} -  Ra\, \nu\, T =-\frac{1}{\rho}\frac{\partial p}{\partial y}+\frac{\partial }{\partial x}\left(\nu \frac{\partial  v}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  v}{\partial y } \right) \tag{3}$$
$$  u \frac{\partial T}{\partial x}+v \frac{\partial T}{\partial y}=\frac{\partial }{\partial x}\left( \frac{\partial  T}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\frac{\partial  T}{\partial y } \right)\tag{4}$$

where $K = k/\rho\,  C_p$ is the diffusivity of the material.

Here `Ra` is the Raliegh number, which  is a dimensionless number associated with buoyancy-driven flow. 

## Application: Spray position for improved nasal drug delivery

In [this paper](https://www.nature.com/articles/s41598-020-66716-0), the authors modeled spray position with a coupled system of static Navier-Stokes equations and transport equations for Lagrangian tracking. The governing equations are

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0 \tag{1}$$
$$u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}=- \frac{1}{\rho}\frac{\partial p}{\partial x}+\frac{\partial }{\partial x}\left( \nu \frac{\partial  u}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  u}{\partial y } \right) + f_1\tag{2}$$
$$u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}  =-\frac{1}{\rho}\frac{\partial p}{\partial y}+\frac{\partial }{\partial x}\left(\nu \frac{\partial  v}{\partial x } \right)+ \frac{\partial }{\partial y}\left(\nu \frac{\partial  v}{\partial y } \right) + f_2\tag{3}$$
$$\frac{\partial w_1}{\partial t} = \kappa_1(u - w_1) + q_1 \tag{5}$$
$$\frac{\partial w_2}{\partial t} = \kappa_2(v - w_2) + q_2 \tag{6}$$


Here $w_1$ and $w_2$ are the droplet velocity. $f_1$ and $f_2$ are  accelerations induced by different body forces. We discretize the transport equation using an implicit finite difference scheme: 

$$\left(\frac{1}{\Delta t}+\kappa_1\right)w_1^{n+1} =\left(\frac{1}{\Delta t}\right)w_1^{n} + \kappa_1 u + q_1^{n+1}$$
$$\left(\frac{1}{\Delta t}+\kappa_2\right)w_2^{n+1} =\left(\frac{1}{\Delta t}\right)w_2^{n} + \kappa_2 v + q_2^{n+1}$$

The computational graph for this coupled system is as follows:

