# Mixed Finite Element Methods for Linear Viscoelasticity 

## Introduction 

One classical approach to linear isotropic elasticity is the displacement-based discretization for 

$$\text{div}(2\mu\varepsilon(u) + \lambda \text{div}\, u\; I) = g$$

Typically $u$ is discretized using continuous piecewise vector polynomials. This method yields accurate approximation for $u$. However, in practice

* The stress is usually the quantity of primary physical interest and pure displacement methods will yield stress approximations of lower order accuracy. 
* The method performs poorly in the incompressible and nearly incompressible case, i.e., $\lambda \rightarrow \infty$.

An alternative approach, the stress-based formulation, addresses this problem by considering a mixed formulation. However, the main obstacle is to construct a stable pair of finite element spaces for symmetric tensors. 

Instead of imposing the symmetry condition on the finite element space directly, we enforce the condition weakly by adding an additional equation and an associated Lagrange multiplier. 

## Mathematical Formulation
Consider the linear viscoelastic Maxwell model 

$$\begin{aligned}A_1\dot \sigma + A_0 \sigma &= \varepsilon(\dot u)\\ \text{div}\,\sigma + \rho f &= \rho\ddot u\end{aligned}\tag{1}$$

Here $\sigma$ is the stress tensor, and $A_0$, $A_1$ are fourth-order material tensors. We introduce the velocity vector 

$$v = \dot u$$

and the rotation of the velocity vector

$$\rho = \nabla v - \varepsilon(v)$$

Let $M$, $V$, and $K$ be the space of matrices, vectors, and skew symmetric matrices on $\Omega$, then the weak form for Eq. 1 is:

Find $(\sigma, v, \rho)\in H(\text{div}, \Omega; M) \times L^2(\Omega; V) \times L^2(\Omega; K)$, such that for all $(\tau, w, \eta)\in H(\text{div}, \Omega; M) \times L^2(\Omega; V) \times L^2(\Omega; K)$

$$\begin{aligned}(A_0\sigma, \tau) + (A_1\dot \sigma, \tau) & + (v, \text{div}\,\tau) &+ (\rho, \tau) &= (\tau \mathbf{n}, v)_{\Gamma_D}\\ (w, \text{div}\,\sigma) &-(\rho \dot v, w) &&=(-\rho f, w)\\ (\sigma, \eta) & & &=0\end{aligned}$$

Here $\Gamma_D$ is the Dirichlet boundary condition for the velocity $v$. Note that if we have traction boundary condition 

$$\sigma \mathbf{n} = g, \quad x\in \Gamma_N\subset \partial\Omega$$

The condition is part of the Dirichlet boundaries for $\sigma$. 

After discretization, this leads to a DAE

$$D_1 \dot y + D_0 y = F$$

where 

$$D_1 = \begin{bmatrix}\mathbf{A}_1 & 0 & 0 \\ 0 & \mathbf{B} & 0 \\
0 & 0 & 0\end{bmatrix}, \quad D_0 = \begin{bmatrix}\mathbf{A}_0 & \mathbf{B} & \mathbf{C}\\ \mathbf{B}^T & 0 & 0 \\ \mathbf{C}^T &0 &0\end{bmatrix}$$

This DAE can be solved using [`TR_BDF2`](https://kailaix.github.io/ADCME.jl/dev/api/#ADCME.TR_BDF2) in ADCME. 

## Linear Viscoelasticity Model

We consider the 2D Maxwell material model here

$$\dot \sigma_{ij} + \frac{\mu}{\eta} \left( \sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij} \right) = 2\mu \dot \epsilon_{ij} + \lambda \dot\epsilon_{kk}\delta_{ij}\tag{2}$$

We will convert Eq. 2 to the form in Eq. 1. To this end, consider two linear operator 

$$I: \sigma \mapsto \sigma, \quad T: \sigma \mapsto (\text{tr}\,\sigma) I$$

In the Voigt notation, these two linear operators have the matrix form 

$$T = \begin{bmatrix}1 & 1 & 0\\ 1 & 1 &0 \\ 0 & 0 & 0\end{bmatrix}, \quad I = \begin{bmatrix}1 & 0 & 0\\ 0 & 1 &0 \\ 0 & 0 & 1\end{bmatrix}$$

The following formulas can be easily derived:

$$T^2 = 2T, (I+\alpha T)^{-1} = I + \beta T, \beta = -\frac{\alpha}{1+2\alpha}$$

Therefore, Eq. 2 can be rewritten as 

$$\dot \sigma + \frac\mu\eta \sigma - \frac{\mu}{3\eta} T\sigma = 2\mu \dot\varepsilon + \lambda T\dot \varepsilon$$

which leads to 

$$\dot\varepsilon = a\sigma + b T\sigma$$

Here 

$$a = \frac{1}{2\mu} + \frac{1}{2\eta}, \quad b = \beta \left(\frac{1}{2\mu} + \frac{1}{2\eta}\right) - \frac{1}{6\eta} - \frac{\beta}{3\eta}, \quad \beta = -\frac{\lambda}{2(\lambda + \mu)}$$

In AdFem, $(a\sigma + b T\sigma, \tau)$ can be calculated using [`compute_fem_bdm_mass_matrix`](@ref). 

## Example 