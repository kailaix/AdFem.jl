for idx in 1 2 3 4 5
do 
for noise in 0.0 0.01 0.03 0.05 0.075 0.1
do 
    julia inverse.jl $noise &
done 
done 
wait 