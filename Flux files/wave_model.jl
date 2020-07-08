using Flux
using Plots
using LinearAlgebra
using Printf
using ProgressMeter
include("Adam_optimise.jl")
#=
 solve the 1D scalar wave equation Ω = [0,1] with

               uₜₜ - c²uₓₓ = 0       (x, t) ∈ Ω×[0,T]
                u(x, 0) = η(x)          x∈Ω
                uₜ(x, 0) = γ(x)         x∈Ω
                u(0, t) = g₀(t)         t>0
                u(1, t) = g₁(t)         t>0
=#

in = 2                  # size of input
out = 1                 # size of output
#α = Float32(0.01)      # learning rate
M = 100                 # number of mini-batches to run
batch_size = 32         # size of each mini-batch
#epochs = 5             # number of passes throguh the training set
z = Float32.(0:0.001:1) # grid for evaluating the trained network
c = Float32(1.0)        # transport velocity

# initial condition with exact solution u(x) = sin(2π(x-t))
η(x) = Float32(sin(2π*x))
γ(x) = Float32(-2π*cos(2π * x))

# boundary condition at x = 0 and x = 1
g₀(t) = Float32(-sin(2π *t))
g₁(t) = Float32(sin(2π *(1 - t)))

# construct a trainable dense neural network with three layers
#θ = Dict(:W1 => Float32.(rand(10,   in)), :b1 => Float32.(rand(10)),
#         :W2 => Float32.(rand(15,   10)), :b2 => Float32.(rand(15)),
#         :W3 => Float32.(rand(out,  15)), :b3 => Float32.(rand(out))
#         )
W1 = Float32.(rand(2,in))
b1 = Float32.(rand(2))

W2 = Float32.(rand(2,2))
b2 = Float32.(rand(2))

W3 = Float32.(rand(out,2))
b3 = Float32.(rand(out))

# set nerual net parameters
θ = params(W1, b1, W2, b2, W3, b3)

#=
u(x, θ) = sum( θ[:W3] * tanh.(
               θ[:W2] * tanh.(
               θ[:W1] * x .+ θ[:b1] ) .+ θ[:b2] ) .+ θ[:b3] )
=#
u(x, W1, b1, W2, b2, W3, b3) = sum( W3 * tanh.(
                                    W2 * tanh.(
                                    W1 * x .+ b1 ) .+ b2 ) .+ b3 )

∇u = (x, W1, b1, W2, b2, W3, b3)-> gradient(
                             x -> u(x, W1, b1, W2, b2, W3, b3), x)[1]

∇₂u = (x, W1, b1, W2, b2, W3, b3)-> gradient(
                             x -> sum(∇u(x, W1, b1, W2, b2, W3, b3)), x)[1]

∇θ = gradient(θ) do
    sum(∇₂u(x, W1, b1, W2, b2, W3, b3))
end

#=
#=
Define a physics informed neural net
            f := uₜₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
function F(x, W1, b1, W2, b2, W3, b3)

end
function f(x, θ)
    uₓ = ∇(x, θ)[1]
    uₜ = ∇(x, θ)[2]

    return uₜ - c*uₓ .+ Float32(4*π.*cos(2*π.*(x[1] - x[2])))
end

function f2(x, θ)
    uₓ = ∇(x, θ)[1]
    uₜ = ∇(x, θ)[2]

    return uₜ + c*uₓ
end

#=
Define a cost function such that minimization enforces initial and boundary
conditions as well as the differential condition uₜ-cuₓ = 0
=#
function cost(s,  α)
    sum(abs.(f2(s[1], α)).^2 +
        abs.(f(s[1], α)).^2 +                    # enforce structure of the PDE
        abs.(u(s[2], α) .- η.(s[2][1])).^2 .+        # initial displacement
        abs.(∇(s[3], α)[2] .- γ.(s[3][1])).^2)# .+        # initial velocity
    #    abs.(u(s[4], α) .- g₀.(s[4][2])).^2 .+       # b.c at x=0
    #    abs.(u(s[5], α) .- g₁.(s[5][2])).^2  )       # b.c at x=1
end

# compile Adam type and optimization function on small data set
#Adam_update!(χ, Adam(χ, χ), χ, 1)

# set weight an biases as Flux parameters
#ps = params(W1, b1, W2, b2, W3, b3)

# initialize Adam objects to store optimization parameters
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

        x̊ = Float32.(rand(0:0.01:1, 2, 1))              # random x∈   Ω×[0,T]
        x̂ = Float32.([rand(0:0.01:1, 1, 1)[1]; 0])      # random x∈   Ω×{0}
        ẋ = Float32.([rand(0:0.01:1, 1, 1)[1]; 0])      # random x∈   Ω×{0}
        x₀ = Float32.([0.0; rand(0:0.01:1.0, 1, 1)[1]])  # random x∈ {0}×[0,T]
        x₁ = Float32.([1.0; rand(0:0.01:1.0, 1, 1)[1]])  # random x∈ {1}×[0,T]

        x = [x̊, x̂, ẋ, x₀, x₁]

        ∇u = gradient(θ -> cost(x, θ), θ)[1]
        #=
        I don't think Flux is capable of handling the computation of ∇u.
        Instead, we define a Monte Carlo method for fast computation of
        second derivatives.
        =#
        # Adam optimisation
        Adam_update!(θ[:W1], W₁, ∇u[:W1], i)
        Adam_update!(θ[:b1], b₁, ∇u[:b1], i)
        Adam_update!(θ[:W2], W₂, ∇u[:W2], i)
        Adam_update!(θ[:b2], b₂, ∇u[:b2], i)
        Adam_update!(θ[:W3], W₃, ∇u[:W3], i)
        Adam_update!(θ[:b3], b₃, ∇u[:b3], i)

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

uexact(x, t) = sin(2π.*(x .- t))
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

    plot!(xfine, uexact.(xfine[:], t[i]), label="exact")

    p2 = plot(xfine, err.(xfine[:], t[i]), ylims = (-0.02, 0.02), label="error", color=:red)

    p = plot(p1, p2, layout = (2,1), size=(1000, 750))

    #plot!(xfine, uexact.(xfine[:] .+ c*t[i]), label="exact")
    display(p)
end
=#
