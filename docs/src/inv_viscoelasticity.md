#  Viscoelasticity Inversion


## Viscoelastic model based with Maxwell material:

- Momentum balance equation:

$$\sigma_{ij,j} + \rho f_i = \rho \ddot u_i$$

- Constitive law:

$$\dot \sigma_{ij} + \frac{\mu}{\eta} \left( \sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij} \right) = 2\mu \dot \varepsilon_{ij} + \lambda \dot\varepsilon_{kk}\delta_{ij}$$

- The model consists of two layers of differnt vicosity:

![](./assets/visco/viscoelasticity_true.png)


## Forward simulation:

![](./assets/visco/visco_time.png)


## Inversion based on horizontal displacement ($x$) on the surface.

| True model                                     | Inverted result                                   | 
| -----------------------------------------------| --------------------------------------------------| 
| ![](./assets/visco/viscoelasticity_true.png)   | ![](./assets/visco/viscoelasticity_result.png)    |


- Iterations during inversion
![](./assets/visco/inv_viscoelasticity.gif)