# Navier-Stokes equations

The Navier-Stokes equations describe the motion of viscous flow formed by a fluid material. Assume the material has density $\rho$, dynamic viscosity $\mu$, velocity $\bf u$, pressure $p$, and body accelerations $\bf g$, the Navier-Stokes equations include the the continuity equation, which describes the conservation of mass:

$$\frac{\partial \rho}{\partial t} + \sum_i \frac{\partial \rho u_i}{\partial x_i} = 0,$$

and the momentum equation, which describes the conservation of momentum in each spatial direction:

$$\frac{\partial \rho u_i}{\partial t} + \sum_j \frac{\partial \rho u_i u_j}{\partial x_j} = \sum_j \frac{\partial \tau_{ij}}{\partial x_j} - \frac{\partial p }{\partial x_i}+\rho g_i, \quad \forall i,$$

where the stress tensor $\tau_{ij}$ is defined as
$$\tau_{ij} = \mu \left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) - \frac{2}{3} \mu \delta_{ij} \frac{\partial u_j}{\partial x_j}.$$

# Navier-Stokes equations for incompressible flow

We assume the fluid material is incompressible, i.e., has constant density $\rho$, and we denote its the kinematic viscosity as $\nu=\mu / \rho$.
The incompressible Navier-Stokes equations in the absence of body accelerations, and in two spatial dimensions, are given by

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0 \tag{1}$$
$$\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}+v \frac{\partial u}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}\right) \tag{2}$$
$$\frac{\partial v}{\partial t}+u \frac{\partial v}{\partial x}+v \frac{\partial v}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} v}{\partial x^{2}}+\frac{\partial^{2} v}{\partial y^{2}}\right) \tag{3}$$

where (1) is the continuity equation and (2)-(3) are the momentum equations.

## The implicit scheme

### Step 1:

The first step is to solve a nonlinear equation

$$\frac{\partial (u+\Delta u) }{\partial t}+(u+\Delta u) \frac{\partial (u+\Delta u)}{\partial x}+(v+\Delta v) \frac{\partial (u+\Delta u)}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} (u+\Delta u)}{\partial x^{2}}+\frac{\partial^{2} (u+\Delta u)}{\partial y^{2}}\right) \tag{4}$$

$$\frac{\partial (v+\Delta v)}{\partial t}+(u+\Delta u) \frac{\partial (v+\Delta v)}{\partial x}+(v+\Delta v) \frac{\partial (v+\Delta v)}{\partial y}=-\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} (v+\Delta v)}{\partial x^{2}}+\frac{\partial^{2} (v+\Delta v)}{\partial y^{2}}\right) \tag{5}$$

Here the inputs $u, v$ are defined in the finite element space, and $p$ in defined in the finite volume space. We solve for $\Delta u, \Delta v$ using the finite element method by linearizing (4)-(5). 

Let $\delta u'$ denote the finite element basis for $u$, and $\delta v'$ denote the finite element basis for $v$. To derive the weak form of (4)-(5), we multiply both sides of (4)-(5) by $\delta u'$ and $\delta v'$, respectively.

$$\int_\Omega \left( \frac{\partial (u+\Delta u) }{\partial t}+(u+\Delta u) \frac{\partial (u+\Delta u)}{\partial x}+(v+\Delta v) \frac{\partial (u+\Delta u)}{\partial y} , \delta u'\right) d\Omega=\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} (u+\Delta u)}{\partial x^{2}}+\frac{\partial^{2} (u+\Delta u)}{\partial y^{2}}\right), \delta u'\right) d\Omega\\
\int_\Omega \left( \frac{\partial (v+\Delta v)}{\partial t}+(u+\Delta u) \frac{\partial (v+\Delta v)}{\partial x}+(v+\Delta v) \frac{\partial (v+\Delta v)}{\partial y}, \delta v'\right) d\Omega=\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} (v+\Delta v)}{\partial x^{2}}+\frac{\partial^{2} (v+\Delta v)}{\partial y^{2}}\right), \delta v'\right) d\Omega$$


We use a backward Euler's method to discretize the equation in time, i.e., 

$$\frac{\partial (u + \Delta u)}{\partial t} \approx \frac{\Delta u}{\Delta t}$$

Then we have the following formula 

$$\int_\Omega \left( \frac{ \Delta u }{\Delta t}+(u+\Delta u) \frac{\partial (u+\Delta u)}{\partial x}+(v+\Delta v) \frac{\partial (u+\Delta u)}{\partial y} , \delta u'\right)d\Omega =\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial x}+\nu\left(\frac{\partial^{2} (u+\Delta u)}{\partial x^{2}}+\frac{\partial^{2} (u+\Delta u)}{\partial y^{2}}\right), \delta u'\right) d\Omega \\
\int_\Omega \left( \frac{\Delta v}{\Delta t}+(u+\Delta u) \frac{\partial (v+\Delta v)}{\partial x}+(v+\Delta v) \frac{\partial (v+\Delta v)}{\partial y}, \delta v'\right) d\Omega=\int_\Omega \left( -\frac{1}{\rho} \frac{\partial p}{\partial y}+\nu\left(\frac{\partial^{2} (v+\Delta v)}{\partial x^{2}}+\frac{\partial^{2} (v+\Delta v)}{\partial y^{2}}\right), \delta v'\right)d\Omega$$

By ignoring the nonlinear term in the weak form, we finally have the following bilinear and linear forms in the weak formulation 

$$\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix} \qquad \begin{bmatrix}F_1\\ F_2\end{bmatrix}$$

Here

$$A_{11} = \frac{1}{\Delta t}\left( \Delta u, \delta u' \right) + \left(  \frac{\partial u}{\partial x}\Delta u, \delta u' \right) + \left( u \frac{\partial \Delta u}{\partial x}, \delta u' \right) + \left( v \frac{\partial \Delta u}{\partial y}, \delta u' \right) + \nu \left( \nabla (\Delta u), \nabla (\delta u') \right)$$

$$A_{12} = \left(\frac{\partial u}{\partial y} \Delta v, \delta u' \right)$$

$$A_{21} = \left(\frac{\partial v}{\partial x} \Delta u, \delta v' \right)$$

$$A_{22} = \frac{1}{\Delta t}\left( \Delta v, \delta v' \right) + \left(  \frac{\partial v}{\partial y}\Delta v, \delta v' \right) + \left(  v\frac{\partial \Delta v}{\partial y}, \delta v' \right) + \left( u \frac{\partial \Delta v}{\partial x}, \delta v' \right) + \nu \left( \nabla (\Delta v), \nabla (\delta v') \right)$$

$$F_1 = \frac{1}{\rho} \left( p, \frac{\partial \delta u'}{\partial x}\right) - \nu (\nabla u, \nabla \delta u') - \left( u\frac{\partial u}{\partial x}, \delta u' \right) -  \left( v\frac{\partial u}{\partial y}, \delta u' \right)$$

$$F_2 = \frac{1}{\rho}  \left( p, \frac{\partial \delta v'}{\partial y}\right)- \nu (\nabla v, \nabla \delta v') - \left( v\frac{\partial v}{\partial y}, \delta v' \right) -  \left( u\frac{\partial v}{\partial x}, \delta v' \right)$$

### Step 2:

We solve the following Poisson equation for the pressure correction term $\Delta p$:

$$\frac{\Delta t}{\rho} \left( \frac{\partial^2 \Delta p}{\partial x^2 } +  \frac{\partial^2 \Delta p}{\partial y^2 } \right) =\frac{\partial u^{*}}{\partial x} + \frac{\partial u^{*}}{\partial y}$$

where the input is obtained by $u^{*} = u + \Delta u, v^{*} = v + \Delta v$ and defined in the finite element method, and we solve for $\Delta p$ defined in the finite volume method.

Then, we can compute the pressure at the next discrete time point

$$p_{new} = p+ \Delta p$$


### Step 3:

Finally, we obtain the velocity field at the next discrete time point by solving the following equation using the finite element method:

$$u_{new} = u^{*} - \frac{\Delta t}{\rho} \frac{\partial \Delta p }{\partial x}$$
$$v_{new} = v^{*} - \frac{\Delta t}{\rho}  \frac{\partial \Delta  p}{\partial y}$$


## Example

We consider the (scaled) Taylor–Green vortex as an example to solve the system of equations (1)-(3).

$$u(x,y,t)=\cos(2\pi x)\sin(2\pi y)\exp(-8\pi^2 \nu t)$$

$$v(x,y,t)=-\sin(2\pi x)\cos(2\pi y)\exp(-8\pi^2 \nu t)$$

$$p(x,y,t)=-\frac{\rho}{4}\left( \cos(4\pi x) + \cos(4\pi y) \right)\exp(-16\pi^2 \nu t)$$


```julia
using ADCME
using PoreFlow
using PyPlot
using SparseArrays

# grid setup
m = 30
n = 30
h = 1/n
bc = bcedge("all", m, n, h)
bd = bcnode("all", m, n, h)
bd_2d = [bd; (m+1)*(n+1) .+ bd]

# time step setup
t = 0;
t_final = 0.01;
NT = 20;
dt = t_final/NT;

# physical constants
ρ = 1
μ = 1
ν = μ / ρ

# pre-compute constant matrices
mass_mat = constant(compute_fem_mass_matrix1(m, n, h))
mass_mat_2d = constant(compute_fem_mass_matrix(m, n, h))
mass_mat_2d_bdry, _ = fem_impose_Dirichlet_boundary_condition(mass_mat_2d, bd, m, n, h) # 2D boundary condition
laplace_mat = constant(compute_fem_laplace_matrix1(m, n, h))
laplace_mat_2d = constant(compute_fem_laplace_matrix(m, n, h))
interact_mat = constant(compute_interaction_matrix(m, n, h))
tpfa_mat_bdry, _ = compute_fvm_tpfa_matrix(ones(m*n), bc, zeros(size(bc,1)), m, n, h)
tpfa_mat_bdry = constant(tpfa_mat_bdry)

# exact solutions
function u1_exact(x1,x2,t)
    cos(2*pi*x1) * sin(2*pi*x2) * exp(-8*pi*pi*ν*t)
end

function u2_exact(x1,x2,t)
    -sin(2*pi*x1) * cos(2*pi*x2) * exp(-8*pi*pi*ν*t)
end

function p_exact(x1, x2, t, ρ)
    -ρ/4 * (cos(4*pi*x1) + cos(4*pi*x2)) * exp(-16*pi*pi*ν*t)
end

function step1(U, p0, Source = missing)
    Source = coalesce(Source, zeros(2*(m+1)*(n+1)))
    u0 = U[1:(m+1)*(n+1)]
    v0 = U[(m+1)*(n+1)+1:end]
    u0_gauss = fem_to_gauss_points(u0, m, n, h)
    v0_gauss = fem_to_gauss_points(v0, m, n, h)

    gradu = eval_grad_on_gauss_pts1(u0, m, n, h) # du/dx = gradu[:,1], du/dy = gradu[:,2]
    gradv = eval_grad_on_gauss_pts1(v0, m, n, h) # dv/dx = gradv[:,1], dv/dy = gradv[:,2]

    M1 = mass_mat
    M2 = compute_fem_mass_matrix1(gradu[:,1], m, n, h)
    M3 = compute_fem_advection_matrix1(u0_gauss, v0_gauss, m, n, h)
    M4 = laplace_mat
    A11 = 1/dt * M1 + M2 + M3 + ν * M4

    A12 = compute_fem_mass_matrix1(gradu[:,2], m, n, h)

    A21 = compute_fem_mass_matrix1(gradv[:,1], m, n, h)

    M2 = compute_fem_mass_matrix1(gradv[:,2], m, n, h) 
    A22 = 1/dt * M1 + M2 + M3 + ν * M4    # M1, M3, M4 are same as A11

    A = [A11 A12
        A21 A22]

    grad_p = compute_interaction_term(p0, m, n, h) # weak form of [dp/dx; dp/dy] on fem points

    s1 = u0_gauss .* gradu[:,1] + v0_gauss .* gradu[:,2]
    s2 = u0_gauss .* gradv[:,1] + v0_gauss .* gradv[:,2]
    b3 = compute_fem_source_term(s1, s2, m, n, h)

    F = Source + 1/ρ * grad_p - ν * laplace_mat_2d * [u0;v0] - b3

    A, _ = fem_impose_Dirichlet_boundary_condition(A, bd, m, n, h)
    F = scatter_update(F, bd_2d, zeros(length(bd_2d)))

    sol = A \ F
    return sol
end

function step2(u0)
    rhs = ρ / dt * interact_mat * u0
    sol = tpfa_mat_bdry \ rhs
    return sol
end

function step3(u0, dp)
    grad_dp = - compute_interaction_term(dp, m, n, h)
    rhs = mass_mat_2d * u0 - dt/ρ * grad_dp
    rhs = scatter_update(rhs, bd_2d, zeros(length(bd_2d)))
    sol = mass_mat_2d_bdry \ rhs
    return sol
end

# input: U length 2(m+1)(n+1)
# input: p length mn
function solve_ns_one_step(U, p)
    dU = step1(U, p)
    U_int = U + dU
    dp = step2(U_int)
    p_new = p + dp
    U_new = step3(U_int, dp)
    return U_new, p_new
end

function condition(i, velo_arr, p_arr)
    i <= NT + 1
end

function body(i, velo_arr, p_arr)
    velo = read(velo_arr, i-1)
    pres = read(p_arr, i-1)
    op = tf.print("i=",i)
    i = bind(i, op)
    velo_new, pres_new = solve_ns_one_step(velo, pres)
    velo_arr = write(velo_arr, i, velo_new)
    p_arr = write(p_arr, i, pres_new)
    return i+1, velo_arr, p_arr
end

# fem nodes
xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u0 = @.  u1_exact(x, y, 0.0)
v0 = @.  u2_exact(x, y, 0.0)
velo_arr = TensorArray(NT+1)
velo_arr = write(velo_arr, 1, [u0; v0])

# fvm nodes
xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @.  p_exact(x, y, 0.0, ρ)
p_arr = TensorArray(NT+1)
p_arr = write(p_arr, 1, p0)

i = constant(2, dtype=Int32)

_, velo, p = while_loop(condition, body, [i, velo_arr, p_arr])
velo = set_shape(stack(velo), (NT+1, 2*(m+1)*(n+1)))
p = set_shape(stack(p), (NT+1, m*n))

sess = Session(); init(sess)
output = run(sess, [velo, p])
out_v = output[1]
out_p = output[2]


figure(figsize=(18,12))
subplot(231)
visualize_scalar_on_fem_points(out_v[1, 1:(1+m)*(1+n)], m, n, h)
title("initial velocity in x direction")
subplot(232)
visualize_scalar_on_fem_points(out_v[1, (1+m)*(1+n)+1:end], m, n, h)
title("initial velocity in y direction")
subplot(233)
visualize_scalar_on_fvm_points(out_p[1, :], m, n, h)
title("initial pressure")
subplot(234) 
visualize_scalar_on_fem_points(out_v[NT+1, 1:(1+m)*(1+n)], m, n, h)
title("final velocity in x direction")
subplot(235) 
visualize_scalar_on_fem_points(out_v[NT+1, (1+m)*(1+n)+1:end], m, n, h)
title("final velocity in y direction")
subplot(236)
visualize_scalar_on_fvm_points(out_p[NT+1, :], m, n, h)
title("final pressure")

```