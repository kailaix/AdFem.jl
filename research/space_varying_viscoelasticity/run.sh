for stepsize in 1 2 3 5
do 
    srun julia viscoelasticity.jl $stepsize &
done 