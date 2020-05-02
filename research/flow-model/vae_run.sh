for sigma in 0.01 0.05 0.1 0.2 0.5
do 
    srun julia vae.jl $sigma &
done
wait