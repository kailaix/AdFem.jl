# Coupled Geomechanics and Multiphase Flow

The main focus of this section is to describe the coupled system of geomechanics and multiphase flow equations. 



## Governing Equations

The governing equation of coupled geomechanics and multiphase flow can be described in terms of solid and fluid equations [^geomechanics].

[^geomechanics]: Wan, Jing. *Stabilized finite element methods for coupled geomechanics and multiphase flow*. Diss. stanford university, 2003. 

**Equations for the Solid**

We assume that the solid density is constant, and therefore, *the mass balance equation of the deforming porous medium* is

$$\frac{\partial}{\partial t} (1-\phi) + \nabla\cdot(1-\phi)\mathbf{v}_s = 0 \Leftrightarrow \frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{v}_s \phi) = \nabla \cdot \mathbf{v}_s \tag{1}$$

The solid velocity $\mathbf{v}_s$ is given by  


$$\mathbf{v}_s = \frac{d\mathbf{u}}{dt}$$ 


and therefore we have 


$$\nabla \cdot \mathbf{v}_s =\frac{d}{dt} \nabla  \cdot \mathbf{u} = \frac{\partial \varepsilon_{vol}}{\partial t}$$


where the volumetric strain 


$$\varepsilon_{vol} = \nabla \cdot \mathbf{u} $$


It can be shown that Equation 1 leads to the *variation of porosity*:

$$\boxed{\phi = 1-(1-\phi_0)\exp(-\varepsilon_{vol}) } \tag{2}$$

**Equations for the Fluids**

The *mass balance equations of multiphase multicomponent fluid* are given by

$$\frac{\partial }{{\partial t}}(\phi {S_i}{\rho _i}) + \nabla  \cdot ({\rho _i}{\mathbf{v}_{is}}) = {\rho _i}{q_i}, \quad i = 1,2 \tag{3}$$

The *linear momentum balance equation* is given by 

$${\mathbf{v}_{is}} =  - \frac{{K{k_{ri}(S_i)}}}{{{\tilde{\mu}_i}}}(\nabla {P_i} - g{\rho _i}\nabla Z), \quad i=1,2 \tag{4}$$

Here, $K$ is the permeability tensor, but in our case we assume it is a space varying scalar value. $k_{ri}(S_i)$ is a function of $S_i$, and typically the higher the saturation, the easier the corresponding phase is to flow. $\tilde \mu_i$ is the viscosity. $Z$ is the depth cordinate, $\rho_i$ is the density, $\phi$ is the porosity, $q_i$ is the source, $P_i$ is the fluid pressure and $g$ is the velocity constant. 

$\mathbf{v}_{is}$ is the relative velocity of the phase $i$ with respect to $\mathbf{v}_s$. 

**Fluid and Mechanics Coupling**

$$\nabla \cdot {\sigma}' - \nabla \left( S_1P_1 + S_2P_2 \right) + \mathbf{f} = 0\tag{5}$$

**Constitutive Relation**

The constitutive relation connects $\sigma'$ and the displacement $\mathbf{u}$. For example, the linear elastic relation is expressed as 

$$\sigma' = \lambda \mathbf{I}\nabla \cdot \mathbf{u} + 2\mu \varepsilon \tag{6}$$

Here, the strain is the Cauchy strain

$$\varepsilon = \frac{1}{2}(\nabla \mathbf{u} + (\nabla \mathbf{u})^T)$$



## Numerical Scheme

We use an iterative algorithm to solve the coupled equation; namely, we alternatively solve the mechanics equation and flow equation. 

![image-20200313003554865](./assets/visco/scheme.png)

**Fluid Equation**

We define the fluid potential 

$$\Psi_i = P_i - \rho_i gZ$$

and the capillary potential 

$$\Psi_c = \Psi_1 - \Psi_2 = P_1 -P_2 - (\rho_1-\rho_2)gZ \approx - (\rho_1-\rho_2)gZ$$

Define mobilities 

$$m_i(S_i) = \frac{k_{ri}(S_i)}{\tilde\mu_i}, i=1,2\quad m_t = m_1 + m_2$$

We have the following formula from Equations 3-4:

$$-\nabla (m_tK\nabla \Psi_2) = \nabla \cdot(m_1 K\nabla \Psi_c) - \frac{\partial \phi}{\partial t} + q_1 + q_2 \tag{7}$$

We can solve for $\Psi_2$ using a Poisson solver. 

Next, we have from Equations 3-4

$$\phi\frac{\partial S_2}{\partial t} + S_2 \frac{\partial\phi}{\partial t} + \nabla \cdot (-K m_2 \nabla \Psi_2) = q_2 + q_1 \frac{m_2}{m_1} \tag{8}$$

Note we have an extra term $q_1 \frac{m_2}{m_1}$ to account for the ignored capillary pressure $P_1-P_2$. 

Equation 8 is a nonlinear equation in $S_2$ ($m_2$ is defined in terms of $S_2=1-S_1$) and requires a Newton-Raphson solver. 

**Solid Equation**

Upon solving the fluid equation, we obtain $S_1, S_2, \Psi_2$. We can use $\Psi_2$ to estimate the fluid pressure $P_1$ and $P_2$. Use Equations 5 and 6, we solve for $\mathbf{u}$ using

$$\int_\Omega \sigma' :\delta \varepsilon \mathrm{d} x + \int_\Omega (S_1P_1+S_2P_2)\delta \varepsilon_v \mathrm{d}x = 0$$

Here $\varepsilon_v = \varepsilon_{xx} + \varepsilon_{yy} = u_x + u_y$. 



