"""
    readMesh(gmshFile::String)

Reads a `gmsh` file and extracts element, coordinates and boundaries.
"""
function Mesh(gmshfile::String)
    cnt = read(gmshfile, String)
    cnt = replace(cnt, "\r"=>"")
    r = r"\$MeshFormat\n(\S*).*\n\$EndMeshFormat"s
    version = match(r, cnt)[1]
    println("Gmsh file version ... $version")

    r = r"\$Nodes\n(.*)\n\$EndNodes"s
    nodes = match(r, cnt)[1]
    nodes = split(nodes,'\n')
    nodes = [parse.(Float64, split(x)) for x in nodes]
    nodes = filter(x->length(x)==3,nodes)
    nodes = hcat(nodes...)'[:,1:2]
    println("Nodes ... $(size(nodes,1))")

    r = r"\$Elements\n(.*)\n\$EndElements"s
    elems = match(r, cnt)[1]
    elems = split(elems,'\n')
    elems = [parse.(Int64, split(x)) for x in elems]
    elems = filter(x->length(x)==4,elems)
    elems = hcat(elems...)'[2:end,2:4]
    println("Elements ... $(size(elems,1))")

    println("Preprocessed nodes ... $(size(nodes,1))")
    println("Preprocessed elements ... $(size(elems,1))")



    return Mesh(nodes, elems)
end