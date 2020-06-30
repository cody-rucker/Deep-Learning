using Flux
using LinearAlgebra
using ReverseDiff
using Plots

#=
 using a simple neural net, the goal is to find its appropriate derivative
 with respect to its input arguments. We want to do this in a way that does not
 interfere with Flux tracking so that these derivatives can be used in a cost
 function and be differentiable with respect to the neural net parameters
 W and b.
=#

ins = 1                 # size of input
outs = 1                # size of output
t = [2.3]

W1 = rand(5, 1)
b1 = rand(5)
layer1(x, W, b) = W .* x .+ b

W2 = rand(1,5)
b2 = rand(1)
layer2(x, W, b) = W .* x .+ b

model(x, W1, b1, W2, b2) = sum(layer2(tanh.(layer1(x, W1, b1)), W2, b2))

function f(t, W1, b1, W2, b2)
    u = model(t, W1, b1, W2, b2)
    du = gradient(() -> model(t, W1, b1, W2, b2), params(t))
    f = du[t][1] + 3u
end

u(t, W₁, b₁, W₂, b₂) = model(t, W₁, b₁, W₂, b₂)[1]

cost(t, W₁, b₁, W₂, b₂) = 1/40000 * abs(f(t, W₁, b₁, W₂, b₂)[1])^2 + 1/40000 * abs(model([0.0], W₁, b₁, W₂, b₂) - 3)^2

ps = params(W1, b1, W2, b2)
α = 4
z = 0:0.01:1
M = 40000
N = 20
z1 = [0.0]
Z1 = zeros(length(z), N +1)
k = 1

for i = 1:M
    t = rand(0:0.01: 1,1)

    grad = gradient(ps) do
        cost(t, W1, b1, W2, b2)
    end

    for j = 1:length(ps)
        ps[j] .-= α .* grad[ps[j]]
    end

    #@show cost(t, W1, b1, W2, b2)
end
#    if i%(M/N) == 0
#                for j = 1:length(z)
#                        z1 .= z[j]
#                        Z1[j, k] = model(z1, W1, b1, W2, b2)
#                end
#                k += 1
#        end
#    end


exact(t) = 3exp(-3t)
Z = zeros(length(z))
for i = 1:length(z)
    Z[i] = model([z[i]], W1, b1, W2, b2)
end

err(t) = exact(t) - model([t], W1, b1, W2, b2)

plot(z, exact.(z), label="exact")
plot!(z, Z, label="network")


#=
parameters = [W1]
q = [0.0]
for j in parameters
    q .= ReverseDiff.gradient([t][1]) do j
        cost(t, j, b1, W2, b2)
    end
end
=#
# feed-forward network u just containing sine for now
#=u = Chain(Dense(1, 20, tanh),
          Dense(20, 15, tanh),
          Dense(15, 1))=#
#u(x) = sum(û(x))

#du(x) =sum( ReverseDiff.gradient(x) do z
#    sum(u(z))
#end)

#du(x) = gradient(u, x)[1]
#ddu(x) = gradient(du, x)[1]

#=
function f(t)
    y = sum(u(t))
    dy = gradient(() -> u(t)[1], params(t))
    f = dy[t][1] - y
end

cost(t) = abs(f(t))^2

grad = gradient(() -> cost(t), params(u))
=#

#=
f(x) = cos(x)           # function to be "taught"

function loss(x)
    ŷ = du(x)
    sum((f.(x) .- ŷ).^2)
end

function teach(u)


    α = 0.1             # learning rate
    num_iter = 90000    # number of iterations

    for i = 1:num_iter
        ξ = rand(-2π:0.01:2π, 1, 1)

        # compute grad w.r.t weights and biases
        gs = gradient(() -> loss(ξ), params(u))

        # gradient descent
        for i = 1:length(u)-1
            u[i].W .-= α .* gs[u[i].W]
            u[i].b .-= α .* gs[u[i].b]
        end
    end
end

function qr(t)

    qr = 2+t
end
=#
