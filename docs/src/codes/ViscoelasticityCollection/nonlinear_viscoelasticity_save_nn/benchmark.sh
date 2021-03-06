for i in 1 2 
do 
for width in 1 5 10 20 40
do 
for depth in 1 3 5 10
do 
for activation in tanh relu selu elu sigmoid 
do 
srun julia viscoelasticity_nn_arch.jl $i $width $depth $activation & 
done
done
done 
done 
wait

