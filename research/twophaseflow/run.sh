for noise in 0.0 0.01 0.05 0.1
do 
    srun julia linear_invert.jl $noise & 
    srun julia visco_eta_invert.jl $noise & 
done 
wait 