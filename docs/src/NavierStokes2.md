

# Navier-Stokes equations for Incompressible Fluid

The incompressible Navier-Stokes equations in the absence of body force, and in two spatial dimensions, are given by

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0 \\
\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}\right) \\
\frac{\partial v}{\partial t}+u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}}\right)$$

The first of the above equation represents the continuity equation and the other two represent the momentum equations.



The implicit scheme

Step 1:

The first step is to solve a nonlinear equation
$$\frac{\partial (u+\Delta u) }{\partial t}+(u+\Delta u) \frac{\partial (u+\Delta u)}{\partial x}+(v+\Delta v) \frac{\partial (u+\Delta u)}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} (u+\Delta u)}{\partial x^{2}}+\frac{\partial^{2} (u+\Delta u)}{\partial y^{2}}\right) \\
\frac{\partial (v+\Delta v)}{\partial t}+(u+\Delta u) \frac{\partial (v+\Delta v)}{\partial x}+(v+\Delta v) \frac{\partial (v+\Delta v)}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} (v+\Delta v)}{\partial x^{2}}+\frac{\partial^{2} (v+\Delta v)}{\partial y^{2}}\right) \tag{1}$$

Here the inputs $u, v$ are defined in the finite element space, and $p$ in defined in the finite volume space. We solve for $\Delta u, \Delta v$ using the finite element method by linearizing Equation 1. 

Let $\delta u'$ denote the finite element basis for $u$, and $\delta v'$ denote the finite element basis for $v$. To derive the weak form of Equation 1, we multiply both sides of Equation 1 by $\delta u'$ and $\delta v'$, respectively.

$$\int_\Omega \left( \frac{\partial (u+\Delta u) }{\partial t}+(u+\Delta u) \frac{\partial (u+\Delta u)}{\partial x}+(v+\Delta v) \frac{\partial (u+\Delta u)}{\partial y} , \delta u'\right) d\Omega=\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} (u+\Delta u)}{\partial x^{2}}+\frac{\partial^{2} (u+\Delta u)}{\partial y^{2}}\right), \delta u'\right) d\Omega\\
\int_\Omega \left( \frac{\partial (v+\Delta v)}{\partial t}+(u+\Delta u) \frac{\partial (v+\Delta v)}{\partial x}+(v+\Delta v) \frac{\partial (v+\Delta v)}{\partial y}, \delta v'\right) d\Omega=\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} (v+\Delta v)}{\partial x^{2}}+\frac{\partial^{2} (v+\Delta v)}{\partial y^{2}}\right), \delta v'\right) d\Omega$$


We use a backward Euler's method to discretize the equation in time, i.e., 

$$\frac{\partial (u + \Delta u)}{\partial t} \approx \frac{\Delta u}{\Delta t}$$

Then we have the following formula 

$$\int_\Omega \left( \frac{ \Delta u }{\Delta t}+(u+\Delta u) \frac{\partial (u+\Delta u)}{\partial x}+(v+\Delta v) \frac{\partial (u+\Delta u)}{\partial y} , \delta u'\right)d\Omega =\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} (u+\Delta u)}{\partial x^{2}}+\frac{\partial^{2} (u+\Delta u)}{\partial y^{2}}\right), \delta u'\right) d\Omega \\
\int_\Omega \left( \frac{\Delta v}{\Delta t}+(u+\Delta u) \frac{\partial (v+\Delta v)}{\partial x}+(v+\Delta v) \frac{\partial (v+\Delta v)}{\partial y}, \delta v'\right) d\Omega=\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} (v+\Delta v)}{\partial x^{2}}+\frac{\partial^{2} (v+\Delta v)}{\partial y^{2}}\right), \delta v'\right)d\Omega$$

By ignoring the nonlinear term in the weak form, we finally have the following bilinear and linear forms in the weak formulation 

$$\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix} \qquad \begin{bmatrix}F_1\\ F_2\end{bmatrix}$$

Here

$$A_{11} = \frac{1}{\Delta t}\left( \Delta u, \delta u' \right) + \left(  \frac{\partial u}{\partial x}\Delta u, \delta u' \right) + \left( u \frac{\partial \Delta u}{\partial x}, \delta u' \right) + \left( v \frac{\partial \Delta u}{\partial y}, \delta u' \right) + \nu \left( \nabla (\Delta u), \nabla (\delta u') \right)$$

$$A_{12} = \left(\frac{\partial u}{\partial y} \Delta v, \delta u' \right)$$

$$A_{21} = \left(\frac{\partial v}{\partial x} \Delta u, \delta v' \right)$$

$$A_{22} = \frac{1}{\Delta t}\left( \Delta v, \delta v' \right) + \left(  \frac{\partial v}{\partial y}\Delta v, \delta v' \right) + \left(  v\frac{\partial \Delta v}{\partial y}, \delta v' \right) + \left( u \frac{\partial \Delta v}{\partial x}, \delta v' \right) + \nu \left( \nabla (\Delta v), \nabla (\delta v') \right)$$

$$F_1 = \frac{1}{\rho} \left( p, \frac{\partial \delta u'}{\partial x}\right) - \nu (\nabla u, \nabla \delta u') - \left( u\frac{\partial u}{\partial x}, \delta u' \right) -  \left( v\frac{\partial u}{\partial y}, \delta u' \right)$$

$$F_2 = \frac{1}{\rho}  \left( p, \frac{\partial \delta v'}{\partial y}\right)- \nu (\nabla v, \nabla \delta v') - \left( v\frac{\partial v}{\partial y}, \delta v' \right) -  \left( u\frac{\partial v}{\partial x}, \delta v' \right)$$

Step 2:

Solve the following Poisson equation for $\Delta p$

$$\frac{\Delta t}{\rho} \left( \frac{\partial^2 \Delta p}{\partial x^2 } +  \frac{\partial^2 \Delta p}{\partial y^2 } \right) =\frac{\partial u^{*}}{\partial x} + \frac{\partial u^{*}}{\partial y}$$

where the input is obtained by $u^{*} = u + \Delta u, v^{*} = v + \Delta v$ and defined in the finite element method, and we solve for the pressure correction term $\Delta p$ defined in the finite volume method.

Then, we can compute the pressure for the next discrete time point

$$p_{new} = p+ \Delta p$$


Step 3:

We solve for the velocity field for the next discrete time point using the finite element method

$$u_{new} = u^{*} - \frac{\Delta t}{\rho} \frac{\partial \Delta p }{\partial x}$$

$$v_{new} = v^{*} - \frac{\Delta t}{\rho}  \frac{\partial \Delta  p}{\partial y}$$



