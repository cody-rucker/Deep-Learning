using Flux
using Plots
using LinearAlgebra
include("Adam_optimise.jl")

#=
mutable struct Adam
  θ::AbstractArray{Float32} # Parameter array
  ∇::AbstractArray{Float32}                # Gradient function
  m::AbstractArray{Float32}     # First moment
  v::AbstractArray{Float32}     # Second moment
  β₁::Float64                   # Exp. decay first moment
  β₂::Float64                   # Exp. decay second moment
  α::Float64                    # Step size
  ϵ::Float64                  # Epsilon for stability
  t::Int                        # Time step (iteration)
end

function Adam(theta::AbstractArray{Float32}, ∇::AbstractArray{Float32})
  m   = zeros(size(theta))
  v   = zeros(size(theta))
  β₁  = 0.9
  β₂  = 0.999
  α   = 0.01
  ϵ   = 1e-8
  t   = 0
  Adam(theta, ∇, m, v, β₁, β₂, α, ϵ, t)
end
=#
#=
 solve a simple linear advection problem on Ω = [0,1] with

               uₜ - cuₓ =
               u(x, 0) = η(x)
               u(0, t) = g₀(t)
=#

ins = 2                 # size of input
outs = 1                # size of output
α = Float32(0.01)       # learning rate
M = 10000                # number of training iterations
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
using Flux.Optimise: update!
using Flux.Optimise: Descent

W₁ = Adam(W1, W1)
b₁ = Adam(b1, b1)

W₂ = Adam(W2, W2)
b₂ = Adam(b2, b2)

W₃ = Adam(W3, W3)
b₃ = Adam(b3, b3)

W = [W₁ b₁ W₂ b₂ W₃ b₃]

# training loop: Adam optimisation
for n = 1:100
    @inbounds for i = 1:32

        x = Float32.(rand(0:0.001:1, 2, 1))              # random x ∈ [0,T]×Ω
        x̂ = Float32.([rand(0:0.001:1, 1, 1)[1]; 0])      # random x∈ {0}×Ω
        x̃ = Float32.([1.0; rand(0:0.001:1.0, 1, 1)[1]])  # random x∈ [0,T]×{1.0}

        ∇u = gradient(ps) do
           cost(x, x̂, x̃, W1, b1, W2, b2, W3, b3)
        end

        Adam_update!(W1, W₁, ∇u[W1], i)
        Adam_update!(b1, b₁, ∇u[b1], i)
        Adam_update!(W2, W₂, ∇u[W2], i)
        Adam_update!(b2, b₂, ∇u[b2], i)
        Adam_update!(W3, W₃, ∇u[W3], i)
        Adam_update!(b3, b₃, ∇u[b3], i)

    #    @inbounds for j = 1:length(ps)

    #        ps[j] .-= α .* ∇u[ps[j]]
    #    end
        @show cost(x, x̂, x̃, W1, b1, W2, b2, W3, b3)


    end
end

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
    p = plot(xfine, Z[:,i], size=(1000, 750), lw=3,
                            legend=:bottomright, label = "network")

    plot!(xfine, sin.(2π.*(xfine[:] .+ c*t[i])), label="exact")
    display(p)
end
