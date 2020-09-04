# attempting to create a NN trained on a continuum of wavespeeds

using Flux
using ForwardDiff
using ProgressMeter
using Plots
include("NNets.jl")

M = 32000   # training iterations
N = 32      # number of hidden neurons

# exact solution
uexact(x,t, c) = exp(-(x-c*t)^2)

# initial condition
η(x) = exp(-x^2)

# boundary condition at x = 0
g₀(t, c) = exp(-(c* t)^2)

# initialize trainable neural net
u = NeuralNet(N, 1, 3)

uₓ = FirstNetDerivative(u, "x₁")
uₜ = FirstNetDerivative(u, "x₂")

# function for constraining differential condition
function f(x, t, c)
    return uₜ(x, t, c) .+ c * uₓ(x, t, c)

end

function cost(x, t,  x̂, t̂, x₀, t₀, λ)
    sum(abs.(f(x,  t, λ)).^2 .+                # enforce structure of the PDE
        abs.(u(x̂,  t̂, λ) .- η.(x̂)).^2 .+       # initial displacement
        abs.(u(x₀, t₀, λ) .- g₀.(t₀, λ)).^2   )   # b.c at x=0
end

# training loop: Adam optimisation
@showprogress "Training..." for n = 1:M

    x = rand(0:0.01:2)      # random (x,t)∈   Ω×[0,T]
    t = rand(0:0.01:3)

    x̂ = rand(0:0.01:3)      # random (x̂,t̂)∈   Ω×{0}
    t̂ = 0.0

    x₀ =0.0                 # random (x₀,t₀)∈ {0}×[0,T]
    t₀ = rand(0:0.01:3.0)

    λ = rand(1:0.01:2)

    ∇u = Flux.gradient(u.π) do
        cost(x, t,  x̂, t̂, x₀, t₀, λ)
    end

    Adam_update(u, ∇u)
end

c = 1:0.25:2
v = 0:0.005:3
xfine = 0:0.01:2
Z = zeros(length(xfine), length(v), length(c))

#err(x, t) = uexact.(x, t, c) - u.(x, t, c)[1]


@inbounds for l = 1:length(c)
    for i = 1:length(v)
        for j = 1:length(xfine)
            Z[j,i,l] = u.(xfine[j], v[i], c[l])[1]
        end
    end
end

@inbounds for i = 1:2:length(v)
    p1 = plot(xfine, Z[:, i, 1], title="λ=1.0", ylims=(-0.25, 1.5), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i], c[1]), label="exact")

    p2 = plot(xfine, Z[:, i, 2],title="λ=1.25", ylims=(-0.25, 1.5), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i], c[2]), label="exact")

    p3 = plot(xfine, Z[:, i, 3],title="λ=1.5", ylims=(-0.25, 1.5), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i], c[3]), label="exact")

    p4 = plot(xfine, Z[:, i, 4], title="λ=1.75", ylims=(-0.25, 1.5), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i], c[4]), label="exact")

    p5 = plot(xfine, Z[:, i, 5], title="λ=2.0", ylims=(-0.25, 1.5), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i], c[5]), label="exact")

    #p2 = plot(xfine, err.(xfine[:], v[i]), ylims = (-0.1, 0.1), label="error", color=:red)

    p = plot(p1, p2, p3, p4, p5, layout = (5,1), size=(1250, 1250))

    display(p)
end
