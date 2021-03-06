for stepsize in  1 
do 
    srun julia viscoelasticity.jl $stepsize &
done 
wait
