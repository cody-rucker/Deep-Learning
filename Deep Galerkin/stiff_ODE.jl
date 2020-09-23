using Flux
using ForwardDiff
using ProgressMeter
using Plots
using DelimitedFiles
include("NNets.jl")

# solve to stiff ODE
#   y' = -100y, t>0, y(0) = 1
# using deep galerkin method

M = 200000   # training iterations
N = 20      # number of hidden neurons

λ=2
# exact solution
yexact(t) = exp(-λ*t)

# initial condition
y₀ = 1

# initialize trainable net
y = NeuralNet(N, 1, 1)

yₜ = FirstNetDerivative(y, "x₁")

# constraint given by the diff eq.
f(t) = yₜ(t) + λ*y(t)

# objective to be minimized
function J(t)
    sum(abs.(f(t)).^2 .+ abs.(y(0) .- y₀).^2)
end




@showprogress "Training..." for n=1:M
    t = rand(0:0.00001:1.0)

    ∇y = Flux.gradient(y.π) do
        J(t)
    end

    Adam_update(y, ∇y)
end

T = 0:0.0001:1.0
Y = zeros(length(T))

for i = 1:length(T)
    Y[i] = y(T[i])[1]
end

err(t) = yexact.(t) .- y.(t)[1]
g(t) = y.(t)[1]

p1 = plot(T, Y[:])
plot!(T, yexact.(T))
p2 = plot(T, err.(T))
p = plot(p1, p2, layout=(2,1))
