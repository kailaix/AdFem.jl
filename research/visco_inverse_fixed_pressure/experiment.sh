for file in nn param linear space_k 
do
    rm -rf $file
    mkdir $file
    cp invdata.mat $file 
    cp coupled_visco_$file.jl $file 
    cd $file
    srun julia coupled_visco_$file.jl &
    cd ..
done
wait 
