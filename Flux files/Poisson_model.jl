using Flux
using Plots
using LinearAlgebra
#=
 solve a simple Poisson problem on Ω = [0,1] with

               -Δu = f
               u(0) = 0
               u(1) = 0
=#

ins = 1                 # size of input
outs = 1                # size of output
α = 0.01                # learning rate
M = 2000               # number of training iterations
z = 0:0.01:1            # grid for evaluating the trained network

c = 1                   # coefficient on y(t) in the ODE

# initial condition
#η(x) = 0.75 * exp(-abs((5*(x[1]-0.5))^2))

# boundary condition at x = 0 and x = 1
#g₀(t) = 0
x₀ = 0
x₁ = 0

# construct a trainable dense neural network with two layers
W1 = rand(5, ins)
b1 = rand(5)
layer1(x, W, b) = W * x .+ b

W2 = rand(outs,5)
b2 = rand(outs)
layer2(x, W, b) = W * x .+ b

u(x, W1, b1, W2, b2) = sum(layer2(σ.(layer1(x, W1, b1)), W2, b2))

#=
Define a physics informed neural net
            f := uₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
F = 1.0
uₓ(x, W1, b1, W2, b2) = sum(gradient(u, x, W1, b1, W2, b2)[1])

function f(x, W1, b1, W2, b2)

    uₓₓ = gradient(() -> uₓ(x, W1, b1, W2, b2), params(x))
    f = - uₓₓ[x][1] - 4π^2 * sin(2π * x[1])
end

#=
Define a cost function such that minimization enforces initial and boundary
conditions as well as the differential condition uₜ-cuₓ = 0
=#

function cost(x, W1, b1, W2, b2)
    (1/M)*(abs(f(x, W1, b1, W2, b2))^2 + abs(uₓ([0.0], W1, b1, W2,b2)-2*π)^2 + abs(uₓ([1.0], W1, b1, W2, b2) - 2π)^2)
end

# set weight an biases as Flux parameters
ps = params(W1, b1, W2, b2)


# training loop
for i = 1:M
    # train model on x∈[0,1], t∈[0,1]
    x = rand(0:0.01:1, 1, 1)

    ∇u = gradient(ps) do
        cost(x, W1, b1, W2, b2)
    end

    # gradient descent
    for j = 1:length(ps)
        ps[j] .-= α .* ∇u[ps[j]]
    end
    @show cost(x, W1, b1, W2, b2)

end


Z = zeros(length(z))

for i = 1:length(z)
    Z[i] = u([z[i]], W1, b1, W2, b2)
end

plot(z, Z)
plot!(z, sin.(2π.*z))
