for i in 1 2
do 
srun julia viscoelasticity.jl $i &
done 
wait 

for i in 1 2
do 
srun julia viscoelasticity_nn.jl $i &
done 
wait 
