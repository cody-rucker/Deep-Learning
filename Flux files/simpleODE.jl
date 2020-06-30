using Flux
using LinearAlgebra
using ReverseDiff
using Plots

#=
solve the ODE y'(t) = λy(t) using a deep neural network
=#

ins = 1                 # size of input
outs = 1                # size of output
α = 1.7                  # learning rate
M = 30000               # number of training iterations
z = 0:0.01:1            # grid for evaluating the trained network

λ = -3                  # coefficient on y(t) in the ODE
y₀ = 3                  # initial condition
exact(t) = y₀*exp(λ*t)  # exact solution

# construct a dense neural network with two layers
W1 = rand(5, ins)
b1 = rand(5)
layer1(x, W, b) = W .* x .+ b

W2 = rand(outs,5)
b2 = rand(outs)
layer2(x, W, b) = W .* x .+ b

model(x, W1, b1, W2, b2) = sum(layer2(σ.(layer1(x, W1, b1)), W2, b2))

# a 'derivative' informed network to impose y'(t) - λy(t) = 0
function f(t, W1, b1, W2, b2)
    u = model(t, W1, b1, W2, b2)
    du = gradient(() -> model(t, W1, b1, W2, b2), params(t))
    f = du[t][1] - λ*u
end

u(t, W₁, b₁, W₂, b₂) = model(t, W₁, b₁, W₂, b₂)[1]

cost(t, W₁, b₁, W₂, b₂) = 1/M *( abs(f(t, W₁, b₁, W₂, b₂)[1])^2 +
                                 abs(u([0.0], W₁, b₁, W₂, b₂) - y₀)^2)

# list of parameters Flux.jl can use to find gradients
ps = params(W1, b1, W2, b2)

# training loop
for i = 1:M
    t = rand(0:0.01: 1,1)

    grad = gradient(ps) do
        cost(t, W1, b1, W2, b2)
    end

    # gradient descent
    for j = 1:length(ps)
        ps[j] .-= α .* grad[ps[j]]
    end

    @show cost(t, W1, b1, W2, b2)
end

Z = zeros(length(z))
for i = 1:length(z)
    Z[i] = model([z[i]], W1, b1, W2, b2)
end

err(t) = exact(t) - model([t], W1, b1, W2, b2)

plot(z, exact.(z), label="exact")
plot!(z, Z, label="network")
