# Mantle Convection


We study the mantle convection and plate motion in this article. We can model this physical phenomenon using a coupled system of velocity and temperature, where the governing equation for the velocity is the balance of linear momentum and incompressibility

$$\begin{aligned}
\text{div} \cdot u &= 0\\ 
-\nabla \cdot \sigma &= Re T e_y
\end{aligned}$$

Here $u$ is the velocity, Re is the Raleigh number, $e_y$ is the unit vector pointing in the $y$ direction, $T$ is the temperature, and $\sigma$ is the stress tensor. 

The conservation of energy is given by 

$$\frac{\partial T}{\partial t} + u \cdot \nabla T - \nabla^2 T = 0$$

We close the system with the following constitutive relation 

$$\sigma = -pI +2\eta_{eff} \epsilon(u)$$

Here $\epsilon(u)$ is the Cauchy strain. The effective viscosity parameter is given by 

$$\eta_{eff} = \eta_{\min} + \min\left( \frac{ \sigma_{yield}}{2\sqrt{\epsilon_{II}}}, \omega \min(\eta_{\max}, \eta) \right)$$

Here $\epsilon_{II} = \frac{1}{2} \epsilon(u) : \epsilon(u)$, and 

$$\eta = Ce^{E(0.5-T)}(\epsilon_{II})^{\frac{1-n}{2n}})$$

We consider a Dirichlet boundary conditions for the temperature on the top and bottom

$$T|_{\partial \Gamma_d} = T_d$$

and a no-flux conditions on the remaining  boundaries

$$\nabla T\cdot n|_{\partial \Omega \backslash \Gamma_d} = 0$$

where $n$ is the unit normal at the boundary. For the Stokes equation, we assume a free-slip mechanical condition on all boundaries

$$u \cdot n = 0\qquad n \times (n\times \sigma n) = 0$$