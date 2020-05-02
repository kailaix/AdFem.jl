for sigma in 0.01 0.05 0.1 0.2 0.5
do 
for N in 500 5000 10000 50000
    srun julia mcmc_hmc.jl $sigma $N &
    srun julia mcmc_rw.jl $sigma $N &
done
wait