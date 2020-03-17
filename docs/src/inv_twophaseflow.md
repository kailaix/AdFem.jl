# Coupled Geomechanics and Multiphase Flow

This section is dedicated to the inverse problem of [Coupled Geomechanics and Multiphase Flow](https://kailaix.github.io/PoreFlow.jl/dev/twophaseflow/). We only consider the parameter inverse problem here, i.e., estimating the Lamé constants and the viscosity parameter from surface horizontal displacement data. We have tried solving the function inverse problem---estimating a nonparametric constitutive relation---using the neural network approach in [the single phase flow problem](https://kailaix.github.io/PoreFlow.jl/dev/coupled_viscoelasticity/), but unfortunately it appears that the limited displacement data are insufficient to train a neural network. 


