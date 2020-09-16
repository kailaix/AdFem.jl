# Numerical experiments

Results of numerical experiments are summarized in the table below. In each case, we assume that the steady-state velocity and temperature data are measured at a sample of locations, represented by a sample of grid points in the finite element method. We solve the optimization problem to minimize the difference between the measured velocity and the computed velocity and temperature based on the current model parameters. In order to model the uncertainty in the measured data, we include a noise level $\epsilon$ when we transform the solution to the forwrad problem into measured data: each data will be multiplied by a random scalar which is uniform on $(1-\epsilon, 1+\epsilon)$.


| grid size | sample size | noise level | number of iterations | exact diffusivity |  predicted diffusivity |
|:---------:|:-----------:|:-----------:|:--------------------:|:-----------------:|:--------------------:|
|  200 by 200 |     10     |      0     |       12             |        2.60475    | 2.60475000000588     |

