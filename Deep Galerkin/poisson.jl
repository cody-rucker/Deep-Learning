using Flux
using ForwardDiff
using ProgressMeter
using Plots
using DelimitedFiles
include("NNets.jl")

M = 20000   # training iterations
N = 7      # number of hidden neurons
ξ = 0.01   # resolution for random sampling of domain

# exact solution
#uexact(x, y) = sin(π * x) + sin(π * y)
uexact(x, y) = exp(-x^2) + exp(-y^2)

# source for manufactured solution uₓₓ + uᵧᵧ + F = 0
#F(x, y) = π^2 * sin(π*x) + π^2 * sin(π * y)
F(x, y) = (4*x^2 - 2) *exp(-x^2) + (4*y^2-2) * exp(-y^2)

# boundary data
#g₀(y) = sin(π * y)
#g₁(y) = sin(π * y)

#h₀(x) = sin(π * x)
#h₁(x) = sin(π * x)

# initialize trainable neural net
u = NeuralNet(N, 1, 2)

uₓₓ = SecondNetDerivative(u, "x₁", "x₁")
uᵧᵧ = SecondNetDerivative(u, "x₂", "x₂")

function f(x, y)
    return -uₓₓ(x, y) .- uᵧᵧ(x, y)
end

function cost(x, y, x̂₀, ŷ₀, x̂₁, ŷ₁, x̄₀, ȳ₀, x̄₁, ȳ₁ )
    sum( abs.(f(x, y) .+ F(x, y)).^2 .+
         abs.(u(x̂₀, ŷ₀) .- uexact(x̂₀,ŷ₀)).^2 .+
         abs.(u(x̂₁, ŷ₁) .- uexact(x̂₁,ŷ₁)).^2 .+
         abs.(u(x̄₀, ȳ₀) .- uexact(x̄₀,ȳ₀)).^2 .+
         abs.(u(x̄₁, ȳ₁) .- uexact(x̄₁,ȳ₁)).^2)
end



@showprogress "Training..." for n = 1:M
    x = rand(0:ξ:1)      # random interior point
    y = rand(0:ξ:1)

    x̂₀ = 0.0                   # random point on x=0
    ŷ₀ = rand(0:ξ:1)

    x̂₁ = 1.0                   # random point on x=1
    ŷ₁ = rand(0:ξ:1)

    x̄₀ = rand(0:ξ:1)      # random point on y=0
    ȳ₀ = 0.0

    x̄₁ = rand(0:ξ:1)      # random point on y=1
    ȳ₁ = 1.0

    ∇u = Flux.gradient(u.π) do
        cost(x, y, x̂₀, ŷ₀, x̂₁, ŷ₁, x̄₀, ȳ₀, x̄₁, ȳ₁ )
    end

    Adam_update(u, ∇u)

    #@show cost(x, y, x̂₀, ŷ₀, x̂₁, ŷ₁, x̄₀, ȳ₀, x̄₁, ȳ₁ )
end

write_params(u)

yfine = 0:0.01:1
xfine = 0:0.01:1
U = zeros(length(xfine), length(yfine))
V = zeros(length(xfine), length(yfine))
E = zeros(length(xfine), length(yfine))
err(x, t) = uexact.(x, t) - u(x, t)[1]# θ)
#plot(legend=true, size=(500,500), xlim=(0,1), ylim=(-1.2,1.2))

@inbounds for i = 1:length(yfine)
    for j = 1:length(xfine)
        U[i,j] = u(xfine[i], yfine[j])[1]
        V[i,j] = uexact(xfine[i], yfine[j])
        E[i,j] = err(xfine[i], yfine[j])
    end
end

p1 = plot(xfine, yfine, U[:, :], st=:contour, title="network", xlabel="x", ylabel="y")
p2 = plot(xfine, yfine, V[:,:], st=:contour, title="exact", xlabel="x", ylabel="y")
p3 = plot(xfine, yfine, E[:,:], st=:contour, title="error", xlabel="x", ylabel="y")
p = plot(p1,p2,p3, layout=(1,3), size=(1250, 750))
