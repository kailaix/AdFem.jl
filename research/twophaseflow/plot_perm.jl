using PyPlot

# Global parameters
K_CONST =  9.869232667160130e-16 * 86400 * 1e3
K = 20.0 .* ones(n,m) # millidarcy
K[8:10,:] .= 120.0 

K *= K_CONST
n = 15
m = 30
h = 30.0 # meter

x = (1:m)*h .- 0.5h
y = (1:n)*h .- 0.5h
pcolormesh(x, y, K, rasterized=true)
plot([3*h], [9*h], ">r")
plot([28*h], [9*h], "<g")
plot(x, 0.45h*ones(m), "v")
gca().invert_yaxis()
xlabel("x")
ylabel("y")
colorbar()
axis("scaled")
gca().set(frame_on=false)
savefig("data/setting.png")
savefig("data/setting.pdf")


