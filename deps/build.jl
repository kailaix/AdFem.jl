using ADCME

rm("build", force=true, recursive=true)
mkdir("$(@__DIR__)/build")
cd("build")
ADCME.cmake()
ADCME.make()
cd("..")


