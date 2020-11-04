using AdFem

function plot_velo_pres_temp_cond(iter) 

    S = run(sess, S_computed)
    u_out, v_out, p_out, T_out = S[1:nnode], 
                                 S[ndof+1:ndof+nnode], 
                                 S[2*ndof+1:2*ndof+nelem],
                                 S[2*ndof+nelem+1:2*ndof+nelem+nnode]

    u_obs, v_obs, p_obs, T_obs = S_data[1:nnode], 
                                 S_data[ndof+1:ndof+nnode], 
                                 S_data[2*ndof+1:2*ndof+nelem],
                                 S_data[2*ndof+nelem+1:2*ndof+nelem+nnode]


    figure(figsize=(15,4))
    subplot(131)
    title("exact x velocity")
    visualize_scalar_on_fem_points(u_obs .* u_std, mesh)
    subplot(132)
    title("predicted x velocity")
    visualize_scalar_on_fem_points(u_out .* u_std, mesh)
    subplot(133)
    title("difference in x velocity")
    visualize_scalar_on_fem_points(u_out .* u_std .- u_obs .* u_std, mesh)
    tight_layout()
    savefig("fn$trialnum/nn_velox$iter.png")

    figure(figsize=(15,4))
    subplot(131)
    title("exact y velocity")
    visualize_scalar_on_fem_points(v_obs .* u_std, mesh)
    subplot(132)
    title("predicted y velocity")
    visualize_scalar_on_fem_points(v_out .* u_std, mesh)
    subplot(133)
    title("difference in y velocity")
    visualize_scalar_on_fem_points(v_out .* u_std .- v_obs .* u_std, mesh)
    tight_layout()
    savefig("fn$trialnum/nn_veloy$iter.png")


    figure(figsize=(15,4))
    subplot(131)
    title("exact pressure")
    visualize_scalar_on_fvm_points(p_obs .* p_std, mesh)
    subplot(132)
    title("predicted pressure")
    visualize_scalar_on_fvm_points(p_out .* p_std, mesh)
    subplot(133)
    title("difference in pressure")
    visualize_scalar_on_fvm_points(p_out .* p_std .- p_obs .* p_std,  mesh)
    tight_layout()
    savefig("fn$trialnum/nn_pres$iter.png")

    
    figure(figsize=(15,4))
    subplot(131)
    title("exact temperature")
    visualize_scalar_on_fem_points(T_obs .* T_infty .+ T_infty, mesh)
    subplot(132)
    title("predicted temperature")
    visualize_scalar_on_fem_points(T_out .* T_infty .+ T_infty, mesh)
    subplot(133)
    title("difference in temperature")
    visualize_scalar_on_fem_points(T_out  .* T_infty .- T_obs .* T_infty, mesh)
    tight_layout()
    savefig("fn$trialnum/nn_temp$iter.png")

    #---------------------------------------------------------------------------
    k_chip_nodes = eval_f_on_fem_pts(k_exact,mesh)[chip_fem_idx_nodes]
    # k_chip_nodes_out = eval_f_on_fem_pts(k_nn,mesh)[chip_fem_idx_nodes]

    # xy = mesh.nodes 
    # xy2 = zeros(mesh.nedge, 2)
    # for i = 1:mesh.nedge
    #     xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
    # end
    # xy = [xy;xy2]

    # x, y = xy[chip_fem_idx_nodes, 1], xy[chip_fem_idx_nodes, 2]
    k_chip_nodes_out = run(sess, k_chip)[1:length(chip_fem_idx_nodes)]
    
    k_all  = k_mold * ones(nnode)
    k_all[chip_fem_idx_nodes] .= k_chip_nodes

    k_all_out = k_mold * ones(nnode)
    k_all_out[chip_fem_idx_nodes] .= k_chip_nodes_out

    

    figure(figsize=(15,4))
    subplot(131)
    visualize_scalar_on_fem_points(k_all, mesh, vmin=minimum(k_chip_nodes), vmax=maximum(k_chip_nodes)); 
    xlim(chip_left, chip_right); ylim(chip_top, chip_bottom)
    title("exact solid conductivity")
    

    subplot(132)
    visualize_scalar_on_fem_points(k_all_out, mesh, vmin=minimum(k_chip_nodes), vmax=maximum(k_chip_nodes)); 
    xlim(chip_left, chip_right); ylim(chip_top, chip_bottom)
    title("predicted solid conductivity")

    subplot(133)
    visualize_scalar_on_fem_points(k_all_out .- k_all, mesh)
    xlim(chip_left, chip_right); ylim(chip_top, chip_bottom)
    title("difference in solid conductivity")

    tight_layout()
    savefig("fn$trialnum/nn_cond$iter.png")

end





function pix_plot_velo_pres_temp_cond(iter) 

    S = run(sess, S_computed)
    u_out, v_out, p_out, T_out = S[1:nnode], 
                                 S[ndof+1:ndof+nnode], 
                                 S[2*ndof+1:2*ndof+nelem],
                                 S[2*ndof+nelem+1:2*ndof+nelem+nnode]

    u_obs, v_obs, p_obs, T_obs = S_data[1:nnode], 
                                 S_data[ndof+1:ndof+nnode], 
                                 S_data[2*ndof+1:2*ndof+nelem],
                                 S_data[2*ndof+nelem+1:2*ndof+nelem+nnode]


    figure(figsize=(15,4))
    subplot(131)
    title("exact x velocity")
    visualize_scalar_on_fem_points(u_obs .* u_std, mesh)
    subplot(132)
    title("predicted x velocity")
    visualize_scalar_on_fem_points(u_out .* u_std, mesh)
    subplot(133)
    title("difference in x velocity")
    visualize_scalar_on_fem_points(u_out .* u_std .- u_obs .* u_std, mesh)
    tight_layout()
    savefig("fn$trialnum/pix_velox$iter.png")

    figure(figsize=(15,4))
    subplot(131)
    title("exact y velocity")
    visualize_scalar_on_fem_points(v_obs .* u_std, mesh)
    subplot(132)
    title("predicted y velocity")
    visualize_scalar_on_fem_points(v_out .* u_std, mesh)
    subplot(133)
    title("difference in y velocity")
    visualize_scalar_on_fem_points(v_out .* u_std .- v_obs .* u_std, mesh)
    tight_layout()
    savefig("fn$trialnum/pix_veloy$iter.png")


    figure(figsize=(15,4))
    subplot(131)
    title("exact pressure")
    visualize_scalar_on_fvm_points(p_obs .* p_std, mesh)
    subplot(132)
    title("predicted pressure")
    visualize_scalar_on_fvm_points(p_out .* p_std, mesh)
    subplot(133)
    title("difference in pressure")
    visualize_scalar_on_fvm_points(p_out .* p_std .- p_obs .* p_std,  mesh)
    tight_layout()
    savefig("fn$trialnum/pix_pres$iter.png")

    
    figure(figsize=(15,4))
    subplot(131)
    title("exact temperature")
    visualize_scalar_on_fem_points(T_obs .* T_infty .+ T_infty, mesh)
    subplot(132)
    title("predicted temperature")
    visualize_scalar_on_fem_points(T_out .* T_infty .+ T_infty, mesh)
    subplot(133)
    title("difference in temperature")
    visualize_scalar_on_fem_points(T_out  .* T_infty .- T_obs .* T_infty, mesh)
    tight_layout()
    savefig("fn$trialnum/pix_temp$iter.png")

    #---------------------------------------------------------------------------
    k_chip_nodes = eval_f_on_fem_pts(k_exact,mesh)[chip_fem_idx_nodes]


    xy = mesh.nodes 
    xy2 = zeros(mesh.nedge, 2)
    for i = 1:mesh.nedge
        xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
    end
    xy = [xy;xy2]

    x, y = xy[chip_fem_idx_nodes, 1], xy[chip_fem_idx_nodes, 2]
    k_chip_nodes_out = run(sess, k_chip[1:length(chip_fem_idx_nodes)])
    
    k_all  = k_mold * ones(nnode)
    k_all[chip_fem_idx_nodes] .= k_chip_nodes

    k_all_out = k_mold * ones(nnode)
    k_all_out[chip_fem_idx_nodes] .= k_chip_nodes_out

    

    figure(figsize=(15,4))
    subplot(131)
    visualize_scalar_on_fem_points(k_all, mesh, vmin=minimum(k_chip_nodes), vmax=maximum(k_chip_nodes)); 
    xlim(chip_left, chip_right); ylim(chip_top, chip_bottom)
    title("exact solid conductivity")
    

    subplot(132)
    visualize_scalar_on_fem_points(k_all_out, mesh, vmin=minimum(k_chip_nodes), vmax=maximum(k_chip_nodes)); 
    xlim(chip_left, chip_right); ylim(chip_top, chip_bottom)
    title("predicted solid conductivity")

    subplot(133)
    visualize_scalar_on_fem_points(k_all_out .- k_all, mesh)
    xlim(chip_left, chip_right); ylim(chip_top, chip_bottom)
    title("difference in solid conductivity")

    tight_layout()
    savefig("fn$trialnum/pix_cond$iter.png")

end