# Viscoelasticity Model for the Earth 

In [previous section](https://kailaix.github.io/PoreFlow.jl/dev/viscoelasticity_earth/), we show how to conduct forward computation of viscoelasticity models for the earth. In this section, we use a gradient-based optimization approach for calibrating the spatial-varying viscoelasticity parameters for the viscoelasticity model. The major function we use is [`ViscoelasticitySolver`](@ref) in [NNFEM](https://github.com/kailaix/NNFEM.jl/) in the NNFEM package.  



In the following examples, we calibrate the viscoelasticity parameters $\eta$ from displacement data on the surface. 

## Strike-slip Fault

| True model                   | Inverted result                  |
| ---------------------------- | -------------------------------- |
| ![](./assets/visco-earth/strikeslip-visco-model.png) | ![](./assets/visco-earth/strikeslip-inv_visco.png) |


We also show the inversion results in each iteration:

```@raw html
<center>
<img src="../assets/visco-earth/strikeslip-inv_visco.gif" width=60%>
</center>
```

## Dip-slip Fault 

| True model                   | Inverted result                  |
| ---------------------------- | -------------------------------- |
| ![](./assets/visco-earth/dipslip-visco-model.png) | ![](./assets/visco-earth/dipslip-inv_visco.png) |


We also show the inversion results in each iteration:

```@raw html
<center>
<img src="../assets/visco-earth/dipslip-inv_visco.gif" width=60%>
</center>
```