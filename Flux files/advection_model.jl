using Flux
using Plots
using LinearAlgebra
#=
 solve a simple linear advection problem on Ω = [0,1] with

               uₜ - cuₓ =
               u(x, 0) = η(x)
               u(0, t) = g₀(t)
=#

ins = 2                 # size of input
outs = 1                # size of output
α = Float32(0.01)       # learning rate
M = 80000                # number of training iterations
z = Float32.(0:0.01:1)  # grid for evaluating the trained network

c = Float32(1.0)        # coefficient on y(t) in the ODE

# initial condition
η(x) = Float32(sin(2π*x))

# boundary condition at x = 0 and x = 1
g₀(t) = Float32(sin(2π *c*t))


# construct a trainable dense neural network with two layers
W1 = Float32.(rand(5, ins))
b1 = Float32.(rand(5))
layer1(x, W, b) = W * x .+ b

W2 = Float32.(rand(10,5))
b2 = Float32.(rand(10))
layer2(x, W, b) = W * x .+ b

W3 = Float32.(rand(outs,10))
b3 = Float32.(rand(outs))
layer3(x, W, b) = W * x .+ b


u(x, W1, b1, W2, b2, W3, b3) = sum(layer3(tanh.(layer2(tanh.(layer1(x, W1, b1)), W2, b2)), W3, b3))

#=
Define a physics informed neural net
            f := uₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
function f(x, y...)
    ∇ = gradient(() ->u(x,y...), params(x))
    uₓ = ∇[x][1]
    uₜ = ∇[x][2]

    f = uₜ - c*uₓ
end

#=
Define a cost function such that minimization enforces initial and boundary
conditions as well as the differential condition uₜ-cuₓ = 0
=#
function cost(x, x̂, x̃, y... )
    sum(abs.(f(x, y...)).^2 +
        abs.(u(x̂, y...) .- η.(x̂[1])).^2 +
        abs.(u(x̃, y...) .- g₀.(x̃[2])).^2)
end

# compile the cost function (and consequently all other functions)
# on a small data set
#χ = Float32.(ones(1,1))
#cost([χ; χ], [χ χ], χ, χ, χ, χ, χ, [χ; χ], [χ; χ])

# set weight an biases as Flux parameters
ps = params(W1, b1, W2, b2, W3, b3)

# training loop: stochastic gradient descent(SGD)
@inbounds for i = 1:M

    x = Float32.(rand(0:0.001:1, 2, 1))
    x̂ = Float32.([rand(0:0.001:1, 1, 1)[1]; 0])
    x̃ = Float32.([1.0; rand(0:0.001:1.0, 1, 1)[1]])

    ∇u = gradient(ps) do
        cost(x, x̂, x̃, W1, b1, W2, b2, W3, b3)
    end

    # gradient descent
    @inbounds for j = 1:length(ps)
        ps[j] .-= α .* ∇u[ps[j]]
    end
    #@show cost(x, W1, b1, W2, b2, W3, b3, x̂, x̃)

end
#=
# training loops: mini-batch SGD
batch_size = 32
# generate minibatch
x = fill(Float32[0.0; 0.0], batch_size, 1)
for i =1:batch_size
    x[i] = rand(0:0.01:1, 2)
end

for i = 1:batch_size
end

=#


t = 0:0.001:1
xfine = 0:0.01:1
Z = zeros(length(z), length(t))

#plot(legend=true, size=(500,500), xlim=(0,1), ylim=(-1.2,1.2))

@inbounds for i = 1:length(t)
    for j = 1:length(xfine)
        Z[j,i] = u([z[j]; t[i]], W1, b1, W2, b2, W3, b3)[1]
    end
end


@inbounds for i = 1:length(t)
    p = plot(xfine, Z[:,i], size=(1000, 750), ylims=(-1.2, 1.2), lw=3,
                            legend=:bottomright, label = "Network")

    plot!(xfine, sin.(2π.*(xfine[:] .+ c*t[i])), label="exact")
    display(p)
end
