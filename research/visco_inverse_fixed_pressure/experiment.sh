for file in nn param simple space_k 
do
    mkdir $file
    cp invdata.jld2 $file 
    cp coupled_visco_$file.jl $file 
done