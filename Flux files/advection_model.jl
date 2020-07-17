using Flux
using ForwardDiff
using ProgressMeter
using Plots
include("Adam_optimise.jl")


M = 1000
batch_size = 64
# initial condition with exact solution u(x) = sin(2π(x-t))
η(x) = sin(2π*x)
γ(x) = -2π*cos(2π * x)

# boundary condition at x = 1
g₁(t) = sin(2π *(1 + t))

# exact solution
uexact(x,t) = exp(-2*t)*sin.(pi*x)

# correction data
F(x, t) = (π^2 - 2) * exp(-2*t) * sin(π*x)

Wₓ = rand(5, 1)
Wₜ = rand(5, 1)
b1 = rand(5)

W2 = rand(5, 5)
b2 = rand(5)

W3 = rand(1, 5)
b3 = rand(1)

θ = Flux.params(Wₓ, Wₜ, b1, W2, b2, W3, b3)

# define a trainable neural network
u(x, t) = sum(W3 * σ.(
              W2 * σ.(
              (Wₓ*x + Wₜ*t) .+ b1) .+ b2) .+ b3)

# sigmoid derivative
dσ(x) =  σ(x) * (1 - σ(x))
d²σ(x) = dσ(x) - 2* dσ(x)*σ(x)

# first-order derivatives
uₓ(x, t) = sum(W3 * (dσ.(W2 * σ.( Wₓ*x .+ Wₜ*t .+ b1) .+ b2) .*
                  (W2 * (dσ.( Wₓ*x .+ Wₜ*t .+ b1) .* Wₓ))))

uₜ(x, t) =sum( W3 * (dσ.(W2 * σ.( Wₓ*x .+ Wₜ*t .+ b1) .+ b2) .*
                  (W2 * (dσ.( Wₓ*x .+ Wₜ*t .+ b1) .* Wₜ))))

# second-order derivatives
#uₓₓ(x, t) = sum(W2 * (d²σ.(Wₓ*x .+ Wₜ*t .+ b1) .* Wₓ .* Wₓ))
#uₜₜ(x, t) = sum(W2 * (d²σ.(Wₓ*x .+ Wₜ*t .+ b1) .* Wₜ .* Wₜ)) #slightly disagrees with autograd...

#=
Define a physics informed neural net
            f := uₜₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
function f(x, t)
    return uₜ(x, t)- uₓ(x, t)
end

function cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
    sum(abs.(f(x,  t)).^2 +                     # enforce structure of the PDE
        abs.(u(x̂,  t̂) .- η.(x̂)).^2 +            # initial displacement
        abs.(u(x₁, t₁) .- g₁.(t₁)).^2   )       # b.c at x=1
end

# initialize Adam objects to store optimization parameters
Wx = Adam(Wₓ, Wₓ)
Wt = Adam(Wₜ, Wₜ)
b₁ = Adam(b1, b1)

W₂ = Adam(W2, W2)
b₂ = Adam(b2, b2)

W₃ = Adam(W3, W3)
b₃ = Adam(b3, b3)


# training loop: Adam optimisation
@showprogress "Training..." for n = 1:M
    sleep(0.1)
    for i = 1:batch_size

        x = rand(0:0.001:1, 1, 1)[1]      # random (x,t)∈   Ω×[0,T]
        t = rand(0:0.001:1, 1, 1)[1]
        x̂ = rand(0:0.001:1, 1, 1)[1]      # random (x,t)∈   Ω×{0}
        t̂ = 0.0
        ẋ = rand(0:0.001:1, 1, 1)[1]      # random (x,t)∈   Ω×{0}
        ṫ = 0.0
        x₀ =0.0                           # random (x,t)∈ {0}×[0,T]
        t₀ = rand(0:0.001:1.0, 1, 1)[1]
        x₁ = 1.0                          # random (x,t)∈ {1}×[0,T]
        t₁ = rand(0:0.001:1.0, 1, 1)[1]

        ∇u = gradient(θ) do
            cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
        end

        # Adam optimisation
        Adam_update!(Wₓ, Wx, ∇u[Wₓ], i)
        Adam_update!(Wₜ, Wt, ∇u[Wₜ], i)
        Adam_update!(b1, b₁, ∇u[b1], i)
        Adam_update!(W2, W₂, ∇u[W2], i)
        Adam_update!(b2, b₂, ∇u[b2], i)
        Adam_update!(W3, W₃, ∇u[W3], i)
        Adam_update!(b3, b₃, ∇u[b3], i)

    # Stochastic gradient descent (SGD)
    #    @inbounds for j = 1:length(θ)
    #        θ[j] .-= α .* ∇u[θ[j]]
    #    end
    #    @show cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
    end
end

v = 0:0.001:1
xfine = 0:0.001:1
Z = zeros(length(xfine), length(v))

uexact(x, t) = sin(2*π*(x+t))
err(x, t) = uexact.(x, t) - u(x, t)

@inbounds for i = 1:length(v)
    for j = 1:length(xfine)
        Z[j,i] = u(xfine[j], v[i])
    end
end

@inbounds for i = 1:2:length(v)
    p1 = plot(xfine, u.(xfine[:], v[i]), size=(1000, 750), ylims=(-1.2, 1.2), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i]), label="exact")

    p2 = plot(xfine, err.(xfine[:], v[i]), ylims = (-1.2, 1.2), label="error", color=:red)

    p = plot(p1, p2, layout = (2,1), size=(1000, 750))

    display(p)
end
