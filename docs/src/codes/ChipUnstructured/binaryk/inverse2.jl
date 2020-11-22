using ADCME
using AdFem

include("../chip_unstructured_solver.jl")
include("../chip_unstructured_geometry.jl")
include("../plot_inverse_one_iter.jl")

trialnum = 19
num_regions = 4

if num_regions == 1
    k_chip_guess = Variable(4.0)
elseif num_regions == 2
    k_chip_guess = Variable([4.0, 1.0])
elseif num_regions == 4
    k_chip_guess = Variable([0.015, 3.0, 1.0, 0.015])
elseif num_regions == 8
    k_chip_guess = Variable([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0])
end


k_mold = 0.014531
k_chip_ref = 2.60475
k_air = 0.64357

function k_exact(x, y)
    if num_regions == 1 || num_regions == 2
        k_mold + (x>0.49 && x<0.5) *  k_chip_ref

    elseif num_regions == 4
        if x<=0.49
            k_mold
        elseif x>0.49 && x<=0.5
            k_chip_ref
        elseif x>0.5 && x<=0.51
            k_air
        else
            k_mold
        end
    end
end

function k_nn(x, y) # exact solution + nn: change to nn solution?

    if num_regions == 1

        if x>0.49 && x<0.5
            constant(k_mold) + k_chip_guess
        else
            constant(k_mold)
        end

    elseif num_regions == 2

        if x>0.49 && x<0.5
            k_chip_guess[1]
        else
            k_chip_guess[2]
        end

    elseif num_regions == 4

        if x<=0.49
            k_chip_guess[1]
        elseif x>0.49 && x<=0.5
            k_chip_guess[2]
        elseif x>0.5 && x<=0.51
            k_chip_guess[3]
        else
            k_chip_guess[4]
        end

    elseif num_regions == 8

        if x<=0.49 && y<= 0.5025
            k_chip_guess[1]
        elseif x>0.49 && x<=0.5 && y<= 0.5025
            k_chip_guess[2]
        elseif x>0.5 && x<=0.51 && y<= 0.5025
            k_chip_guess[3]
        elseif x>0.51 && y<= 0.5025
            k_chip_guess[4]
        elseif x<=0.49 && y> 0.5025
            k_chip_guess[5]
        elseif x>0.49 && x<=0.5 && y> 0.5025
            k_chip_guess[6]
        elseif x>0.5 && x<=0.51 && y> 0.5025
            k_chip_guess[7]
        elseif x>0.51 && y> 0.5025
            k_chip_guess[8]
        end

    end


    
end

xy = mesh.nodes 
xy2 = zeros(mesh.nedge, 2)
for i = 1:mesh.nedge
    xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
end
xy = [xy;xy2]

x, y = xy[chip_fem_idx, 1], xy[chip_fem_idx, 2]
k_chip_exact = constant( @. k_exact(x, y) )
k_chip = stack( @. k_nn(x, y))

nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

NT = 15    # number of iterations for Newton's method

heat_source_fem = zeros(ndof)
heat_source_fem[chip_fem_top_idx] .= power_source
heat_source_gauss = dof_to_gauss_points(heat_source_fem, mesh)

B = constant(compute_interaction_matrix(mesh))
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(ngauss)), mesh))
heat_source = compute_fem_source_term1(constant(heat_source_gauss), mesh)

# apply Dirichlet to velocity and temperature; set left bottom two points to zero to fix rank deficient problem for pressure
bd = [bd; bd .+ ndof; 
      fvm_bd .+ 2*ndof; 
      bd .+ (2*ndof+nelem)]

# add solid region into boundary condition for u, v, p, i.e. exclude solid when solving Navier Stokes
bd = [bd; solid_fem_idx; solid_fem_idx .+ ndof; solid_fvm_idx .+ 2*ndof]

S0 = constant(zeros(nelem+3*ndof))
S = solve_navier_stokes(S0, NT, k_chip)
S_computed = S[end, :]

if num_regions <= 2
    S_data = matread("data.mat")["V"]
else
    S_data = matread("data$num_regions.mat")["V"]
end

loss =  mean((S_computed .- S_data)^2)
loss = loss * 1e10

# ---------------------------------------------------
_loss = Float64[]
cb = (vs, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
    # if mod(iter, 1)==1
    close("all")
    plot_cond(iter, vs[1][1:length(chip_fem_idx_nodes)])
    matwrite("fn$trialnum/loss$iter.mat", Dict("iter"=>iter,"loss"=>_loss, "k_chip"=>vs[1]))
    # end
end


ADCME.options.training.training = placeholder(true)
l = placeholder(rand(20545,))
x = placeholder(rand(5875, 2))

# train the neural network 
opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)
for i = 1:100
    _, loss_ = run(sess, [opt, loss], feed_dict=Dict(l=>S_data, x=>xy))
    @info i, loss_
end

BFGS!(sess, loss, vars = [k_chip], callback = cb)
