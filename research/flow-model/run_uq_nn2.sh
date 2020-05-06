for prior in 1.0 0.1 0.01 0.001
do 
for sigma in 0.0 0.01 0.05 0.1 0.5
do 
    srun julia uq_nn2.jl $sigma $prior &
done
done 
wait 