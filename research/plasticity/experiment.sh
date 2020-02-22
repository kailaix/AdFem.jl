for file in nn simple space_k 
do
    mkdir $file
    cp plasticity.mat $file 
    cp coupled_visco_$file.jl $file 
    cd $file 
    srun julia coupled_visco_$file.jl & 
    cd ..
done
