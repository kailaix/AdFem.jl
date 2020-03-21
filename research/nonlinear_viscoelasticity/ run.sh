for i in 1 2
do 
srun viscoelasticity.jl $i &
done 
wait 

for i in 1 2
do 
srun viscoelasticity_nn.jl $i &
done 
wait 