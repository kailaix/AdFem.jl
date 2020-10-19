using MAT
using PyPlot

SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


loss = matread("sess_figures/ex1/loss_nn/loss1.mat")["L"]

for k = 2:50
    loss_ = matread("sess_figures/ex1/loss_nn/loss$k.mat")["L"]
    global loss = [loss; loss_]
end

close("all"); figure();semilogy(loss); tight_layout()
savefig("sess_figures/ex1/nn/loss.png")

loss2 = matread("sess_figures/ex1/loss_ones/loss1.mat")["L"]

for k = 2:50
    loss_ = matread("sess_figures/ex1/loss_ones/loss$k.mat")["L"]
    global loss2 = [loss2; loss_]
end

close("all"); figure();semilogy(loss2); tight_layout()
savefig("sess_figures/ex1/ones/loss.png")

close("all"); figure();
semilogy(loss, lw=2, label="DNN"); 
semilogy(loss2, lw=2, label="pointwise"); legend();
xlabel("number of iterations");
ylabel("loss")
tight_layout()
savefig("sess_figures/ex1/loss.png")