![](https://travis-ci.com/kailaix/AdFem.jl.svg?token=tCRK4npbxWQNS6KVeBvs&branch=master)![Documentation](https://github.com/kailaix/AdFem.jl/workflows/Documentation/badge.svg)

<p align="center">
<img src="https://github.com/ADCMEMarket/ADCMEImages/blob/master/AdFem/logo.png" width="500"/>
</p>


| Documentation                                                |
| ------------------------------------------------------------ |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kailaix.github.io/AdFem.jl/dev/) |

AdFem is a finite element method open source library for inverse modeling in computational and mathematical engineering. It provides a set of reusable, flexible, and differentiable operators for building scalable and efficient simulators for partial differential equations. 

AdFem is built on [ADCME](https://github.com/kailaix/ADCME.jl), an automatic differentiation library for computational and mathematical engineering. It was originally developed for prototyping inverse modeling algorithms using structured meshes but later evolved into a general and powerful tool with a scalable FEM backend [MFEM](https://mfem.org/). 

`Ad` in AdFem stands for "automatic differentiation" or "adjoint". 

## Forward Computation in AdFem

As an example, we consider solving the Poisson's equation in AdFem

![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/eq1.svg)

Here

![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/eq2.svg)

The weak form for the Poisson's equation is to solve a variational equation 

![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/eq3.svg)

The problem is easily translated in AdFem:

```julia
using AdFem
using PyPlot 

# forward computation
mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))
xy = gauss_nodes(mmesh)
κ = @. sin(xy[:,1]) * (1+xy[:,2]^2) + 1.0
f = 1e5 * @. xy[:,1] + xy[:,2]
K = compute_fem_laplace_matrix1(κ, mmesh)
F = compute_fem_source_term1(f, mmesh)
bdnode = bcnode(mmesh)
K, F = impose_Dirichlet_boundary_conditions(K, F, bdnode, zeros(length(bdnode)))
sol = K\F
```

<p align="center">
<img src="https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/poisson_solution.png" width="500"/>
</p>

The above code shows how to use a linear finite element space to approximate the state variable on a given mesh, define boundary conditions, and construct the linear system. 

## Inverse Modeling

Most functions of AdFem, such as `compute_fem_laplace_matrix1`, `compute_fem_source_term1`, and `impose_Dirichlet_boundary_conditions`, AD-capable, meaning that you can back-propagate gradients from their outputs to inputs. This enables you to conduct inverse modeling without writing extra substantial effort once the forward computation codes are implemented. AdFem constructs a static computational graph for finite element simulators: the computational graph is optimized before executation, and all computations are delegated to efficient C++ kernels. 

Here we use a deep neural network to approximate κ(x) (`fc` is an ADCME function and stands for fully-connected):

```julia
nn_κ = squeeze(fc(xy, [20,20,20,1])) + 1
K = compute_fem_laplace_matrix1(nn_κ, mmesh)
F = compute_fem_source_term1(f, mmesh)
bdnode = bcnode(mmesh)
K, F = impose_Dirichlet_boundary_conditions(K, F, bdnode, zeros(length(bdnode)))
nn_sol = K\F
loss = sum((nn_sol - sol)^2)

sess = Session(); init(sess)
BFGS!(sess, loss)
```

![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/poisson_kappa.png)

## Installation 

AdFem is tested on Unix platform (Linux and Mac). To install the stable release:

```julia
using Pkg
Pkg.add("AdFem")
```

To install the latest version:

```julia
using Pkg 
Pkg.add(PackageSpec(url="https://github.com/kailaix/AdFem.jl", rev="master")) 
```





## Research

AdFem is an open-source package that accompanies ADCME.jl for solving inverse problems involving partial differential equations (PDEs). AdFem provides users a rich collection of operators, which users can use to quickly build finite element/volumn codes for forward computation. More importantly, these operators can back-propagate gradients, and therefore users can calculate the gradients using the ideas of adjoint methods and reverse-mode automatic differention (these two concepts overlap). The advanced physics constrained learning (PCL) approach enables users to back-propagate gradients through iterative and nonlinear solvers efficiently. AdFem offers a flexible interface for experienced researchers to develop their own operators.

Some related research works can be found here:

1. [Physics constrained learning for data-driven inverse modeling from sparse observations](https://arxiv.org/abs/2002.10521)
2. [Solving Inverse Problems in Steady State Navier-Stokes Equations using Deep Neural Networks](https://arxiv.org/abs/2008.13074)
3. [Inverse Modeling of Viscoelasticity Materials using Physics Constrained Learning](https://arxiv.org/abs/2005.04384)


## License

AdFem is licensed under MIT License. See [LICENSE](https://github.com/kailaix/AdFem.jl/blob/master/LICENSE) for details.
