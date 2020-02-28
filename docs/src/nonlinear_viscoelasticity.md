# Nonlinear Viscoelasticity

We consider the nonlinear elasticity. In this case, the viscosity depends on the stress. [^nonlinear viscosity]

[^nonlinear viscosity]: http://www.mate.tue.nl/~piet/inf/trc/pdf/infmamo1d.pdf

![image-20200227153439348](/Users/kailaix/Desktop/PoreFlow.jl/docs/src/assets/nonlinear.png)

The constitutive equations are 
$$
\begin{aligned}
\dot \varepsilon &= \dot \varepsilon_e + \dot \varepsilon_v\\
\sigma &= s + w = E\varepsilon_e + H \varepsilon\\
\dot \varepsilon_v &= \frac{1}{\eta(|s|)}s
\end{aligned}
$$
The high dimensional correspondence is 

$$ k_1s_{ij} + \eta(|s|)\dot s_{ij} = k_1k_2e_{ij} + (k_1+k_2) \eta\dot e_{ij}$$

where

$$e_{ij} = \varepsilon_{ij} - \varepsilon_{kk} \delta_{ij} \qquad \sigma_{ij} = s_{ij} + K\varepsilon_{kk}\delta_{ij}$$

