# Viscoelasticity Model for the Earth 

In [previous section](https://kailaix.github.io/PoreFlow.jl/dev/viscoelasticity_earth/), we show how to conduct forward computation of viscoelasticity models for the earth. In this section, we use a gradient-based optimization (L-BFGS-S) approach for calibrating the spatial-varying viscoelasticity parameters for the viscoelasticity model. The major function we use is [`ViscoelasticitySolver`](@ref) in [NNFEM](https://github.com/kailaix/NNFEM.jl/) in the NNFEM package.  



In the following examples, we calibrate the viscoelasticity parameters $\eta$ from displacement data on the surface. 

## Strike-slip Fault

In this example, we consider a layer model. In each layer, $\eta$ is a constant. The left panel in the following graph shows the ground truth, while the inversion result is shown in the right panel. We can see the inverted $\eta$ is quite accurate after 2000 iterations.  

| True model                   | Inverted result                  |
| ---------------------------- | -------------------------------- |
| ![](./assets/visco-earth/strikeslip-visco-model.png) | ![](./assets/visco-earth/strikeslip-inv_visco.png) |


We show the inversion results in each iteration:

```@raw html
<center>
<img src="../assets/visco-earth/strikeslip-inv_visco.gif" width=60%>
</center>
```


Code:  [antiplane_viscosity_inverse.jl](https://github.com/kailaix/PoreFlow.jl/blob/master/research/earthquake/strikeslip/antiplane_viscosity_inverse.jl)

## Dip-slip Fault 

In this example, we consider a linear viscosity model with an increasing viscosity effect at a deeper depth. Because of the limited observation data (displacement on the surface), we do not expect to calibrate a spatially-varying $\eta$ for each location. Therefore, we reduce the number of optimizable variables b dividing the computational domain into multiple patches. The patch is obtained using K-means algorithm provided by `Clustering.jl`: 

```@raw html
<center>
<img src="../assets/visco-earth/dipslip-patch.png" width=50%>
</center>
```

The true vsicoelasticity parameter distribution is shown in the left panel in the following graph. The right panel shows the inverted result after 200 iterations. We can see that the inverted result is reasonably good. 



| True model                   | Inverted result                  |
| ---------------------------- | -------------------------------- |
| ![](./assets/visco-earth/dipslip-linear_model.png) | ![](./assets/visco-earth/dipslip-inv_visco.png) |


We also show the inversion results in each iteration:

```@raw html
<center>
<img src="../assets/visco-earth/dipslip-inv_visco.gif" width=80%>
</center>
```

Code: [dippingfault_viscosity_forward.jl](https://github.com/kailaix/PoreFlow.jl/blob/master/research/earthquake/dipslip/dippingfault_viscosity_forward.jl), [dippingfault_viscosity_inversion.jl](https://github.com/kailaix/PoreFlow.jl/blob/master/research/earthquake/dipslip/dippingfault_viscosity_inversion.jl), [load_domain_function.jl](https://github.com/kailaix/PoreFlow.jl/blob/master/research/earthquake/dipslip/load_domain_function.jl).