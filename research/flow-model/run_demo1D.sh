for model in arcsin lognormal mixture
do 
for sigma in 0.001 0.01 0.1 1 10
do
    srun julia demo1D.jl $model $sigma &
done 
done 
wait 