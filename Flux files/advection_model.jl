using Flux
using Plots
using LinearAlgebra
using Printf
using ProgressMeter
include("Adam_optimise.jl")
#=
 solve a simple linear advection problem on Ω = [0,1] with

               uₜ - cuₓ = 0       (x, t) ∈ Ω×[0,T]
                u(x, 0) = η(x)          x∈Ω
                u(0, t) = g₀(t)         t>0
=#

ins = 2                 # size of input
outs = 1                # size of output
α = Float32(0.01)       # learning rate
M = 1500                 # number of mini-batches to run
batch_size = 64         # size of each mini-batch
epochs = 5              # number of passes throguh the training set
z = Float32.(0:0.001:1)  # grid for evaluating the trained network
c = Float32(1.0)        # transport velocity

# initial condition
η(x) = Float32(sin(2π*x))
#η(x) = Float32(cos(2*π*(x-0.5)) + 1)
# boundary condition at x = 0 and x = 1
g₀(t) = Float32(sin(2π *c*t))
#g₀(t) = Float32(0.0 * t)
# construct a trainable dense neural network with three layers

θ = Dict(:W1 => Float32.(rand(10,  ins)), :b1 => Float32.(rand(10)),
         :W2 => Float32.(rand(15,   10)), :b2 => Float32.(rand(15)),
         :W3 => Float32.(rand(outs, 15)), :b3 => Float32.(rand(outs))
         )
#=
W1 = Float32.(rand(10, ins))
b1 = Float32.(rand(10))

W2 = Float32.(rand(15,10))
b2 = Float32.(rand(15))

W3 = Float32.(rand(outs,15))
b3 = Float32.(rand(outs))

layer(x, W, b) = W * x .+ b

v(x, W1, b1, W2, b2, W3, b3) = sum(layer(tanh.(layer(tanh.(layer(x, W1, b1)), W2, b2)), W3, b3))
=#
u(x, θ) = sum( θ[:W3] * tanh.(
               θ[:W2] * tanh.(
               θ[:W1] * x .+ θ[:b1] ) .+ θ[:b2] ) .+ θ[:b3] )
#=
Define a physics informed neural net
            f := uₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
function f(x, θ)
    #∇ = gradient(() ->u(x,y...), params(x))
    ∇ = gradient(x -> u(x, θ), x)
    uₓ = ∇[1][1]
    uₜ = ∇[1][2]

    return uₜ - c*uₓ
end

#=
Define a cost function such that minimization enforces initial and boundary
conditions as well as the differential condition uₜ-cuₓ = 0
=#
function cost(x, x̂, x̃, θ)
        abs.(f(x, θ)).^2 +
        abs.(u(x̂, θ) .- η.(x̂[1])).^2 +
        abs.(u(x̃, θ) .- g₀.(x̃[2])).^2
end

# compile the cost function (and consequently all other functions)
# on a small data set
# (does gradient need to be compiled? It came with Flux)
χ = Float32.(ones(1,1))
#gradient(params(χ)) do
#    cost([χ; χ], [χ; χ], [χ; χ], [χ χ], χ, χ, χ, χ, χ )
#end

# compile Adam type and optimization function on small data set
Adam_update!(χ, Adam(χ, χ), χ, 1)

# set weight an biases as Flux parameters
#ps = params(W1, b1, W2, b2, W3, b3)

# initialize Adam objects to store optimization parameters
#=
W₁ = Adam(W1, W1)
b₁ = Adam(b1, b1)

W₂ = Adam(W2, W2)
b₂ = Adam(b2, b2)

W₃ = Adam(W3, W3)
b₃ = Adam(b3, b3)
=#

W₁ = Adam(θ[:W1], θ[:W1])
b₁ = Adam(θ[:b1], θ[:b1])

W₂ = Adam(θ[:W2], θ[:W2])
b₂ = Adam(θ[:b2], θ[:b2])

W₃ = Adam(θ[:W3], θ[:W3])
b₃ = Adam(θ[:b3], θ[:b3])


# training loop: Adam optimisation
 @showprogress "Training..." for n = 1:M
    sleep(0.1)
    #@printf("Training mini-batch number (%d / %d) in epoch (%d / %d)\n", n, M, k, epochs)
    # M mini-batches each of size batch_size
    for i = 1:batch_size

        x = Float32.(rand(0:0.0001:1, 2, 1))              # random x ∈ [0,T]×Ω
        x̂ = Float32.([rand(0:0.0001:1, 1, 1)[1]; 0])      # random x∈ {0}×Ω
        x̃ = Float32.([1.0; rand(0:0.0001:1.0, 1, 1)[1]])  # random x∈ [0,T]×{1.0}

    #    ∇u = gradient(ps) do
    #       cost(x, x̂, x̃, W1, b1, W2, b2, W3, b3)
    #    end

        ∇u = gradient(θ -> cost(x, x̂, x̃, θ), θ)[1]

        # Adam optimisation

        Adam_update!(θ[:W1], W₁, ∇u[:W1], i)
        Adam_update!(θ[:b1], b₁, ∇u[:b1], i)
        Adam_update!(θ[:W2], W₂, ∇u[:W2], i)
        Adam_update!(θ[:b2], b₂, ∇u[:b2], i)
        Adam_update!(θ[:W3], W₃, ∇u[:W3], i)
        Adam_update!(θ[:b3], b₃, ∇u[:b3], i)
#=
        Adam_update!(W1, W₁, ∇u[W1], i)
        Adam_update!(b1, b₁, ∇u[b1], i)
        Adam_update!(W2, W₂, ∇u[W2], i)
        Adam_update!(b2, b₂, ∇u[b2], i)
        Adam_update!(W3, W₃, ∇u[W3], i)
        Adam_update!(b3, b₃, ∇u[b3], i) =#
    #    Adam_update!(W4, W₄, ∇u[W4], i)
    #    Adam_update!(b4, b₄, ∇u[b4], i)

    # Stochastic gradient descent (SGD)
    #    @inbounds for j = 1:length(ps)
    #        ps[j] .-= α .* ∇u[ps[j]]
    #    end
    #    @show cost(x, x̂, x̃, W1, b1, W2, b2, W3, b3)
    end
end


t = 0:0.001:1
xfine = 0:0.001:1
Z = zeros(length(z), length(t))

#=
function uexact(x)
    if x < 0
        return Float32(0)
    elseif x > 1
        return Float32(0)
    else
    return Float32(cos(2*π*(x-0.5)) + 1)
    end
end
=#

uexact(x, t) = sin(2π.*(x .+ c*t))
err(x, t) = uexact.(x, t) - u([x; t],θ)# W1, b1, W2, b2, W3, b3)# θ)
#plot(legend=true, size=(500,500), xlim=(0,1), ylim=(-1.2,1.2))

@inbounds for i = 1:length(t)
    for j = 1:length(xfine)
        Z[j,i] = u([z[j]; t[i]], θ)[1]#θ)[1]
    end
end


@inbounds for i = 1:length(t)
    p1 = plot(xfine, Z[:,i], size=(1000, 750), ylims=(-1.2, 1.2), lw=1.5,
                            legend=:bottomright, label = "network")

    plot!(xfine, sin.(2π.*(xfine[:] .+ c*t[i])), label="exact")

    p2 = plot(xfine, err.(xfine[:], t[i]), ylims = (-0.02, 0.02), label="error", color=:red)

    p = plot(p1, p2, layout = (2,1), size=(1000, 750))

    #plot!(xfine, uexact.(xfine[:] .+ c*t[i]), label="exact")
    display(p)
end
