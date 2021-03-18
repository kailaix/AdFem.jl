# Install & Reproduce AdFem With Docker Guide

An elegant way to easily install AdFem, then reproduce and visualize the demo codes is to use **Docker** + **VSCode-insider** + **VSCode Docker extension** + **VSCode Jupyter extension** + **IJulia**

You can run AdFem demo codes in Jupyter notebook and display the visualization result in it, and here is how it looks:  

![img](https://user-images.githubusercontent.com/47491676/111093187-3d3be200-8573-11eb-82ab-4df78a7e6659.png)


A docker image with Julia 1.5.3 and AdFem installed provided by @Ricahrd-Li is [zhehaoli/julia_adfem](https://hub.docker.com/r/zhehaoli/julia_adfem), and you can just pull and run it. 

Infomation about this docker image: 
```shell
julia> VERSION
v"1.5.3"

(@v1.5) pkg> status
Status `~/.julia/environments/v1.5/Project.toml`
  [07b341a0] ADCME v0.6.7
  [c10abcb9] AdFem v0.1.1
  [7073ff75] IJulia v1.23.2
  [d330b81b] PyPlot v2.9.0
```

### How to run this docker image in VSCode-insider: 
1. Follow [this article from Microsoft](https://devblogs.microsoft.com/python/introducing-the-jupyter-extension-for-vs-code/) and installed VSCode-insider with VSCode Docker extension + VSCode Jupyter extension + IJulia Jupyter kernel.
2. Then start target docker image, and attach to it in VSCode-insider.
3. Open a Jupyter notebook and connect to your Julia kernel(remember to use ``"Notebook: Select Notebook Kernel"`` command to select IJulia kernel), then you will be able to smoothly run and visualize Julia codes in Jupyter notebook in VSCode-insider. 

### Detailed procedure to build your own docker image with AdFem: (provided by @Ricahrd-Li)
1. First, get [Docker Official Images of Julia](https://hub.docker.com/_/julia) with ``docker pull julia`` 
2. Then start the container and get into Julia REPL with ``docker run -it julia_adfem``
3. Then run ``] add AdFem``, which will install dependency packages like ``ADCME`` and compile them. In practice of @Ricahrd-Li, though the version of Julia running here is 1.5.3, the **long-waiting** problem when "Looking for TensorFlow Dynamic Libraries" (mentioned in issue [#13](https://github.com/kailaix/AdFem.jl/issues/13), [kailaix/ADCME.jl#64](https://github.com/kailaix/ADCME.jl/issues/64)) disappears.
    
    **Note**:You may get some compilation errors during ``] add AdFem``, like this one:
    ```shell
    ../CholeskyOp/CholeskyOp.h:1:10: fatal error: 'adept.h' file not found
    #include "adept.h"
            ^~~~~~~~~
    1 error generated.
    ```
    Please check out issue [#13](https://github.com/kailaix/AdFem.jl/issues/13) to solve this error. For other compilation errors, rerun ``use AdFem`` to recompile it and recompilation should solve the rest errors. 
    After installation of ``AdFem`` finishes, you are good to go and share your docker image on docker hub. 

