# Static Linear Elasticity

The governing equation for static linear elasticity is

$$\begin{aligned}
\mathrm{div}\ \sigma(u) &= f(x) & x\in \Omega \\
\sigma(u) &= C\varepsilon(u) \\
u(x) &= u_0(x) & x\in \Gamma_u\\
\sigma(x) n(x) &= t(x) & x\in \Gamma_n
\end{aligned}$$

Here $\varepsilon(u) = \frac{1}{2}(\nabla u + (\nabla u)^T)$ is the Cauchy tensor, $\Gamma_u \cup \Gamma_n = \Omega$, $\Gamma_u \cap \Gamma_n = \emptyset$. The weak formulation is: finding $u$ such that 

$$\int_\Omega \delta \varepsilon(u) : C \varepsilon(u)\mathrm{d} x = \int_{\Gamma_n} t\cdot\delta u \mathrm{d}s - \int_\Omega f\cdot \delta u \mathrm{d}x$$

In this section, we 