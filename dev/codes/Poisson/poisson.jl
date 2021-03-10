using AdFem

# create a uniform 10x10 mesh on [0,1]^2
mmesh = Mesh(10,10,0.1)

# xy is a Nx2 matrix for Gauss point coordinates, 
# where N is the number of Gauss points
xy = gauss_nodes(mmesh)

# fc_init initializes a DNN weights and biases with 3 hidden layers, 20 neurons per layers
# input dimension is 2 (coordinate), and output is 1 (scalar κθ value)
θ = Variable(fc_init([2, 20,20,20,1]))
κ = squeeze(fc(xy, [20,20,20,1], θ))

# construct the coefficient matrix from κ
A = compute_fem_laplace_matrix1(κ, mmesh)

# assume the source function is given by F(x,y) = x + y 
src = xy[:,1] + xy[:,2]
f = compute_fem_source_term1(src, mmesh)

# impose homogenous Dirichlet boundary condition 
A, f = impose_Dirichlet_boundary_conditions(A, f, bcnode(mmesh), zeros(length(bcnode(mmesh))))

# solves the linear system A*u = f 
u = A\f

# Let's generate some random observation data on each FEM node 
uobs = rand(zeros(mmesh.nnode))

loss = sum((u-uobs)^2)

# construct the gradient 
g = gradients(loss, θ) 

sess = Session(); init(sess)

# a one-liner for L-BFGS-B optimization 
BFGS!(sess, loss, method = "L-BFGS-B")


