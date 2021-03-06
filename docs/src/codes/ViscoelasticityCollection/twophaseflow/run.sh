for idx in 1 2 3 4 5 6 7 8 9 10
do 
for noise in 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
do 
    srun julia linear_invert.jl $noise & 
    srun julia visco_eta_invert.jl $noise & 
done 
done 

wait 