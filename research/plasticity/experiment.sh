# for file in nn simple space_k 
# do
#     mkdir $file
#     cp plasticity.mat $file 
#     cp coupled_visco_$file.jl $file 
# done

for file in nn1 nn2 nn3 nn4 nn5
do
    # mkdir $file
    # cp plasticity.mat $file 
    # cp coupled_visco_nn.jl $file 
    cd $file 
    srun julia coupled_visco_nn.jl &
    cd ..
done
