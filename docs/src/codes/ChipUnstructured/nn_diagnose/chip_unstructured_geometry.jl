# geometry setup in domain [0,1]^2
solid_left = 0.45
solid_right = 0.55
solid_top = 0.5
solid_bottom = 0.52

chip_left = 0.48
chip_right = 0.52
chip_top = 0.5
chip_bottom = 0.505

delta=1e-5
delta2=1e-5

filename = "mesh/CHT_2D.stl"
file_format = "stl"
mesh = Mesh(filename, file_format = file_format, degree=2)
mesh = Mesh(mesh.nodes ./ 0.030500000342726707, mesh.elems, -1, 2)
h = 0.1

# m = 100
# n = 100
# h = 1/n
# mesh = Mesh(m, n, h, degree=2)

nnode = mesh.nnode
nedge = mesh.nedge
ndof = mesh.ndof
nelem = mesh.nelem
ngauss = get_ngauss(mesh)

# compute solid indices and chip indices
solid_fem_idx = Array{Int64, 1}([])
solid_fvm_idx = Array{Int64, 1}([])
chip_fem_idx = Array{Int64, 1}([])
# chip_fvm_idx = Array{Int64, 1}([])
chip_fem_top_idx = Array{Int64, 1}([])
bd = Array{Int64, 1}([])
fvm_bd = Array{Int64, 1}([])



for j = 1:nnode
    nodex, nodey = mesh.nodes[j, 1], mesh.nodes[j, 2]
    if nodex >= solid_left-delta2 && nodex <= solid_right+delta2 && nodey >= solid_top-delta2 && nodey <= solid_bottom+delta2
        # print(i, j)
        global solid_fem_idx = [solid_fem_idx; j]
        if nodex >= chip_left-delta2 && nodex <= chip_right+delta2 && nodey >= chip_top-delta2 && nodey <= chip_bottom+delta2
            global chip_fem_idx = [chip_fem_idx; j]
            if nodey <= chip_top+delta2
                global chip_fem_top_idx = [chip_fem_top_idx; j]
            end
        end
    end
    if abs(nodex-0.0) <= delta || abs(nodex-1.0) <= delta || abs(nodey-0.0) <= delta || abs(nodey-1.0) <= delta
        global bd = [bd; j]
    end
end

# save a copy of solid_fem_idx without edges for plotting
chip_fem_idx_nodes = copy(chip_fem_idx)

# fix chip_fem_top_idx
if size(chip_fem_top_idx, 1) == 0
    chip_fem_top_idx = chip_fem_idx
end

for j = 1:nedge
    edgex, edgey = (mesh.nodes[mesh.edges[j, 1], :] .+ mesh.nodes[mesh.edges[j, 2], :]) ./ 2
    if edgex >= solid_left-delta2 && edgex <= solid_right+delta2 && edgey >= solid_top-delta2 && edgey <= solid_bottom+delta2
        # print(i, j)
        global solid_fem_idx = [solid_fem_idx; nnode + j]
        if edgex >= chip_left-delta2 && edgex <= chip_right+delta2 && edgey >= chip_top-delta2 && edgey <= chip_bottom+delta2
            global chip_fem_idx = [chip_fem_idx; nnode + j]
            if edgex >= chip_left-delta2 && edgex <= chip_right+delta2 && edgey >= chip_top-delta2 && edgey <= chip_top+delta2
                global chip_fem_top_idx = [chip_fem_top_idx; nnode + j]
            end
        end
    end
    if abs(edgex-0.0) <= delta || abs(edgex-1.0) <= delta || abs(edgey-0.0) <= delta || abs(edgey-1.0) <= delta
        global bd = [bd; nnode + j]
    end
end

gaussxy = gauss_nodes(mesh)

for i in 1:nelem
    gaussx, gaussy = mean(gaussxy[6*i-5: 6*i, 1]), mean(gaussxy[6*i-5: 6*i, 2])
    if gaussx >= solid_left-delta2 && gaussx <= solid_right+delta2 && gaussy >= solid_top-delta2 && gaussy <= solid_bottom+delta2
        global solid_fvm_idx = [solid_fvm_idx; i]
    end
    if abs(gaussy - 0.0) <= h/2 + delta && ( abs(gaussx - 0.0) <= h + delta || abs(gaussx - 1.0) <= h + delta )
        global fvm_bd = [fvm_bd; i]
    end
end