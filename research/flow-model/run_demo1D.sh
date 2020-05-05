for sigma in 0.001 0.01 0.1 1 10
do
for $dim_z in 1 10 20
do 
for $batch_size in 1 16 32 64
do 
    srun julia demo1D.jl $sigma $dim_z $batch_size &
done 
done
done 
wait 