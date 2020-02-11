#  Viscoelasticity Inversion


## Viscoelastic model based with Maxwell material:

- Momentum balance:

$$\sigma_{ij,j} + \rho f_i = \rho \ddot u_i$$

- Constitutive law:

$$\dot \sigma_{ij} + \frac{\mu}{\eta} \left( \sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij} \right) = 2\mu \dot \varepsilon_{ij} + \lambda \dot\varepsilon_{kk}\delta_{ij}$$

- Boundary conditions:

$$
\begin{aligned}
\bm{\sigma} \mathbf{\hat{n}} &=
\begin{cases}
\mathbf{t} &= 0 & \text{Top or Bottom} \\ 
\mathbf{t} &= [-T, 0] & \text{Right} \\
\end{cases}  \\
u &=0 \text{\hspace{3.1cm} Left} &
\end{aligned}
$$


- The model consists of two layers of differnt vicosity. 

![](./assets/visco/viscoelasticity_true.png)


## Forward simulation:

![](./assets/visco/visco_time.png)


## Inversion based on horizontal displacement $u_x$ on the surface.

| True model                                     | Inverted result                                   | 
| -----------------------------------------------| --------------------------------------------------| 
| ![](./assets/visco/viscoelasticity_true.png)   | ![](./assets/visco/viscoelasticity_result.png)    |


- Iterations during inversion
![](./assets/visco/inv_viscoelasticity.gif)