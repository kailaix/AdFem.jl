for sigma in 0.01 0.05 0.1 0.2 0.5
do
do 
for N in 100000 300000
    srun julia mcmc_hmc.jl $sigma $N &
    srun julia mcmc_rw.jl $sigma $N &
done
done 
wait