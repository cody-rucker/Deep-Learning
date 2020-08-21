using Flux
using ForwardDiff
using ProgressMeter
using Plots
using Flux.Optimise: update!
include("NNets.jl")
include("Adam_optimize.jl")


M = 64000
# initial condition with exact solution u(x) = sin(2π(x-t))
η(x) = exp(-x^2)

# boundary condition at x = 1
g₀(t) = exp(-t^2)

# exact solution
uexact(x,t) = exp(-(x-t)^2)

# initialize trainable neural net
u = NeuralNet2(10, 1)

Wₓ = u.Wₓ
Wₜ = u.Wᵧ
b1 = u.b₁

W2 = u.W₂
b2 = u.b₂

W3 = u.W₃
b3 = u.b₃

θ = params(Wₓ, Wₜ, b1, W2, b2, W3, b3)

# sigmoid derivative
dσ(x) =  σ(x) * (1 - σ(x))
d²σ(x) = dσ(x) - 2* dσ(x)*σ(x)

# first-order derivatives
uₓ(x, t) = sum(W3 * (σ'.(W2 * σ.( Wₓ*x .+ Wₜ*t .+ b1) .+ b2) .*
                  (W2 * (σ'.( Wₓ*x .+ Wₜ*t .+ b1) .* Wₓ))))

uₜ(x, t) =sum( W3 * (σ'.(W2 * σ.( Wₓ*x .+ Wₜ*t .+ b1) .+ b2) .*
                  (W2 * (σ'.( Wₓ*x .+ Wₜ*t .+ b1) .* Wₜ))))

# function for constraining differential condition
function f(x, t)
    return uₜ(x, t) .+ uₓ(x, t)

end

function cost(x, t,  x̂, t̂, x₀, t₀)
    sum(abs.(f(x,  t)).^2 .+                     # enforce structure of the PDE
        abs.(u(x̂,  t̂) .- η.(x̂)).^2 .+            # initial displacement
        abs.(u(x₀, t₀) .- g₀.(t₀)).^2   )       # b.c at x=1
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

    x = rand(0:0.01:2)      # random (x,t)∈   Ω×[0,T]
    t = rand(0:0.01:3)

    x̂ = rand(0:0.01:3)      # random (x̂,t̂)∈   Ω×{0}
    t̂ = 0.0

    x₀ =0.0                 # random (x₀,t₀)∈ {0}×[0,T]
    t₀ = rand(0:0.01:3.0)

    ∇u = Flux.gradient(θ) do
        cost(x, t,  x̂, t̂, x₀, t₀)
    end

    #for p in θ
    #    update!(opt, θ, ∇u)
    #end
    # Adam optimisation
    Adam_update!(Wₓ, Wx, ∇u[Wₓ], n)
    Adam_update!(Wₜ, Wt, ∇u[Wₜ], n)
    Adam_update!(b1, b₁, ∇u[b1], n)
    Adam_update!(W2, W₂, ∇u[W2], n)
    Adam_update!(b2, b₂, ∇u[b2], n)
    Adam_update!(W3, W₃, ∇u[W3], n)
    Adam_update!(b3, b₃, ∇u[b3], n)

# Stochastic gradient descent (SGD)
#    @inbounds for j = 1:length(θ)
#        θ[j] .-= α .* ∇u[θ[j]]
#    end
#    @show cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)

end


v = 0:0.005:3
xfine = 0:0.01:2
Z = zeros(length(xfine), length(v))

err(x, t) = uexact.(x, t) - u.(x, t)[1]


@inbounds for i = 1:length(v)
    for j = 1:length(xfine)
        Z[j,i] = u.(xfine[j], v[i])[1]
    end
end

@inbounds for i = 1:2:length(v)
    p1 = plot(xfine, Z[:, i], size=(1000, 750), ylims=(-0.25, 1.5), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i]), label="exact")

    p2 = plot(xfine, err.(xfine[:], v[i]), ylims = (-0.1, 0.1), label="error", color=:red)

    p = plot(p1, p2, layout = (2,1), size=(1000, 750))

    display(p)
end
