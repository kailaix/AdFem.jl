# Inverse Modeling for Coupled Geomechanics and Single Phase Flow

We have coupled geomechanics and single phase flow in [`Coupled Geomechanics and Single Phase Flow`](@ref). Now we consider the inverse modeling: assuming that the plane stress elasticity matrix $H$ is unknown, we want to estimate $H$ based on the observation data-- the displacement and velocity on a line of sensors.



In the inverse modeling code, we need to replace for loops by `while_loop` syntax. Additionally, we need to using `Variable(H)` to mark $H$ as trainable. We run the `L-BFGS-B` algorithm and obtain the following loss function profile and estimated $H$. 

| u displacement              | v displacement              | Pressure                    |
| --------------------------- | --------------------------- | --------------------------- |
| ![](./assets/disp_u_tf.gif) | ![](./assets/disp_v_tf.gif) | ![](./assets/disp_p_tf.gif) |



[Code](./assets/invpi.jl)



![](./assets/loss.png)

| Initial Guess                                                | Estimated $H$                                                | Reference $H$                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $$\begin{bmatrix}1 &   0 & 0 \\ 0 & 1  & 0 \\ 0    &  0    &  1\\\end{bmatrix}$$ | $$\begin{bmatrix}1.13961  &   0.398872 &    -2.35165\times10^{-6}\\ 0.398842  &  1.13959 &     4.39933\times10^{-6}\\ 3.74498\times10^{-6} & 2.38203\times10^{-6}&   0.740731\\\end{bmatrix}$$ | $$\begin{bmatrix}1.1396 &   0.39886 & 0.0 \\ 0.39886 & 1.1396  & 0.0 \\ 0.0    &  0.0    &  0.740741\\\end{bmatrix}$$ |



