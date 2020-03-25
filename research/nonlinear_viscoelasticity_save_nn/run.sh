for i in 2 
do 
srun julia viscoelasticity_nn.jl $i & 
done 
wait
