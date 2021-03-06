using PyPlot
using Revise
close("all")

function interface(x, y, t, θ, x0, x1, x2, y1, y2, t1, t2)
  
  sigmoid(x, a) = @. 1. / (1. + exp(-(x - a)))
  
  window(x, a1, a2) = sigmoid(x, a1) .* (1. .- sigmoid(x, a2)) 

  # @show x
  # @show x1, x2, y1, y2
  # error()

  vx = window(x, x1, x2) .* window(y, y1, y2) .* window(t, t1, t2)

  vy = zeros(size(vx))
  if length(x) == 1
    if x > x0
      vy =  vx * sin(θ)
      vx = vx * cos(θ)
    end
  else
    vy[x .> x0] = vx[x .> x0] .* sin(θ)
    vx[x .> x0] = vx[x .> x0] .* cos(θ)
  end

  dt = t[2] - t[1]
  ux = cumsum(vx) * dt 
  ax = zeros(size(vx)) 
  ax[2:end] = (vx[2:end] .- vx[1:end-1]) ./ dt
  uy = cumsum(vy) * dt 
  ay = zeros(size(vy)) 
  ay[2:end] = (vy[2:end] .- vy[1:end-1]) ./ dt

  return [ux, uy, vx, vy, ax, ay]
end

θ = π/6
nt = 100
nx = 100
ny = 100
x0 = 40.
t1 = 20.
t2 = 30.
x1 = 30.
x2 = 80.
y1 = 0.
y2 = 5.
# t1 = 1.
# t2 = 100.
# x1 = 1.
# x2 = 100.
# y1 = 1.
# y2 = 100.
t = collect(1.:nt)
# x = collect(1.:nx)
# y = collect(1.:ny)
seg1_x = collect(0:40)
seg1_y = seg1_x .* 0
seg2_x = collect(40.:0.5:60.)
seg2_y = (seg2_x .- 40) .* tan(θ)
segs_x = [seg1_x; seg2_x]
segs_y = [seg1_y; seg2_y]
# seg3_x = collect(40:100)
# seg3_y = seg3_x .* 0
# segs_x = [seg1_x; seg2_x; seg3_x]
# segs_y = [seg1_y; seg2_y; seg3_y]


vx = []
vy = []
ux = []
uy = []
ax = []
ay = []
for (x, y) in zip(segs_x, segs_y)
  (ux_, uy_, vx_, vy_, ax_, ay_) = interface(x, y, t, θ, x0, x1, x2, y1, y2, t1, t2)
  push!(ux, ux_)
  push!(uy, uy_)
  push!(vx, vx_)
  push!(vy, vy_)
  push!(ax, ax_)
  push!(ay, ay_)
end
ux = hcat(ux...)
uy = hcat(uy...)
vx = hcat(vx...)
vy = hcat(vy...)
ax = hcat(ax...)
ay = hcat(ay...)


figure()

it = 25
# for it = 1:nt
# clf()

subplot(311)
quiver(segs_x, segs_y, vx[it,:], vy[it,:], angles="xy", scale=20)
gca().invert_yaxis()

subplot(312)
quiver(segs_x, segs_y, ux[it,:], uy[it,:], angles="xy", scale=100)
gca().invert_yaxis()

subplot(313)
quiver(segs_x, segs_y, ax[it,:], ay[it,:], angles="xy", scale=0.1)
gca().invert_yaxis()

# title("$it")
# show()
# end