# AdFem


![](./assets/logo.png)



| Documentation                                                |
| ------------------------------------------------------------ |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kailaix.github.io/PoreFlow.jl/dev/) |

AdFem is a finite element method open source library for inverse modeling in computational and mathematical engineering. It provides a set of reusable, flexible, and differentiable operators for building scalable and efficient simulators for partial differential equations. 

AdFem is built on [ADCME](https://github.com/kailaix/ADCME.jl), an automatic differentiation library for computational and mathematical engineering. It was originally developed for prototyping inverse modeling algorithms using structured meshes but later evolved into a general and powerful tool with a scalable FEM backend [MFEM](https://mfem.org/). 

`Ad` in AdFem stands for "automatic differentiation" or "adjoint". 