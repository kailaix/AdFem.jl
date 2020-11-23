using MAT
using PyPlot; matplotlib.use("agg")
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

d = matread("summary17.mat")["dist"][1:17]
l = matread("summary17.mat")["loss"][1:17]

d2 = matread("../diagnose1+/summary100.mat")["dist"][1:100]
l2 = matread("../diagnose1+/summary100.mat")["loss"][1:100]

d3 = matread("summary031.mat")["dist"][1:31]
l3 = matread("summary031.mat")["loss"][1:31]


d = [d; d2; d3]
l = [l; l2; l3]

figure()
scatter(d, l, s=100, marker="x", c="#ff0000", zorder=999)
plt.yscale("log")
xlabel("distance")
ylabel("loss")
tight_layout()

savefig("test.png")