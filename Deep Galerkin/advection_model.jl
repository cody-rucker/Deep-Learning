using Flux
using ForwardDiff
using ProgressMeter
using Plots
using DelimitedFiles
include("NNets.jl")

M = 32000   # training iterations
N = 7       # number of hidden neurons
batch_size = 32
# exact solution
uexact(x,t) = exp(-(x-t)^2)

# initial condition
η(x) = [exp(-x^2)]

# boundary condition at x = 0
g₀(t) = [exp(-t^2)]

# initialize trainable neural net
u = NeuralNet(N, 1, 2)

uₓ = FirstNetDerivative(u, "x₁")
uₜ = FirstNetDerivative(u, "x₂")

# function for constraining differential condition
function f(x, t)
    return uₜ(x, t) .+ uₓ(x, t)

end

function cost(x, t,  x̂, t̂, x₀, t₀)
    sum((f.(x,  t)).^2 .+                # enforce structure of the PDE
        (u.(x̂,  t̂) .- η.(x̂)).^2 .+       # initial displacement
        (u.(x₀, t₀) .- g₀.(t₀)).^2   )   # b.c at x=0
end

# training loop: Adam optimisation
@showprogress "Training..." for n = 1:M

    x = rand(0:0.01:2, batch_size)      # random (x,t)∈   Ω×[0,T]
    t = rand(0:0.01:3, batch_size)

    x̂ = rand(0:0.01:3, batch_size)      # random (x̂,t̂)∈   Ω×{0}
    t̂ = zeros(batch_size)

    x₀ =zeros(batch_size)                # random (x₀,t₀)∈ {0}×[0,T]
    t₀ = rand(0:0.01:3.0, batch_size)

    ∇u = Flux.gradient(u.π) do
        cost(x, t,  x̂, t̂, x₀, t₀)[1]
    end

    Adam_update(u, ∇u)
end

write_params(u)

v = 0:0.005:3
xfine = 0:0.01:2
Z = zeros(length(xfine), length(v))

err(x, t) = uexact.(x, t) - u.(x, t)[1]


for i = 1:length(v)
    for j = 1:length(xfine)
        x = rand(0:0.01:1)
        t = rand(0:0.01:1)
        Z[i, j] = err(x, t)
    end
end
max_err = findmax(Z)
"""
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
"""
