# PDE Galleries

Here is a collection of common partial differential equations and how you can solve them using the PoreFlow library. Unless we specify particularly, the computational domain will be $\Omega = [0,1]^2$. The configuration of the computational domain is as follows


```@raw html
<img src="./assets/domain.png" width="60%">
```

We only show the forward modeling, but the inverse modeling is a by-product of the AD-capable implementation!

## Poisson's Equation 

Consider the Poisson's equation 

$$-\Delta u = f \qquad u|_{\partial \Omega} = 0$$

The analytical solution is given by 

$$u(x,y) = \sin \pi x \sin \pi y$$

We have

$$f(x,y) = 2\pi^2 \sin \pi x \sin \pi y$$

```@example
using PyPlot 
using PoreFlow

m = 50; n = 50; h = 1/n 

A = constant(compute_fem_laplace_matrix1(m, n, h))
F = eval_f_on_gauss_pts((x,y)->2π^2*sin(π*x)*sin(π*y), m, n, h)
bd = bcnode("all", m, n, h)
A, _ = fem_impose_Dirichlet_boundary_condition1(A, bd, m, n, h)
rhs = compute_fem_source_term1(F, m, n, h)
rhs[bd] .= 0.0
sol = A\rhs

sess = Session(); init(sess)
S = run(sess, sol)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(S, m, n, h)
title("Computed")
subplot(122)
visualize_scalar_on_fem_points(eval_f_on_fem_pts((x,y)->sin(π*x)*sin(π*y), m, n, h), m, n, h)
title("Reference")

savefig("poisson.png"); nothing # hide
```

![](poisson.png)