using Flux
using ForwardDiff
using ProgressMeter
using Plots
include("NNets.jl")

M = 64000   # training iterations
N = 10      # number of hidden neurons

# exact solution
uexact(x,t) = exp(-(x-t)^2)

# initial condition
η(x) = exp(-x^2)

# boundary condition at x = 0
g₀(t) = exp(-t^2)

# initialize trainable neural net
u = NeuralNet2(N, 1)

Wₓ = u.Wₓ.θ
Wₜ = u.Wᵧ.θ
b1 = u.b₁.θ

W2 = u.W₂.θ
b2 = u.b₂.θ

W3 = u.W₃.θ
b3 = u.b₃.θ

uₓ = FirstNetDerivative(u, "x")
uₜ = FirstNetDerivative(u, "y")

# function for constraining differential condition
function f(x, t)
    return uₜ(x, t) .+ uₓ(x, t)

end

function cost(x, t,  x̂, t̂, x₀, t₀)
    sum(abs.(f(x,  t)).^2 .+                # enforce structure of the PDE
        abs.(u(x̂,  t̂) .- η.(x̂)).^2 .+       # initial displacement
        abs.(u(x₀, t₀) .- g₀.(t₀)).^2   )   # b.c at x=0
end

# training loop: Adam optimisation
@showprogress "Training..." for n = 1:M

    x = rand(0:0.01:2)      # random (x,t)∈   Ω×[0,T]
    t = rand(0:0.01:3)

    x̂ = rand(0:0.01:3)      # random (x̂,t̂)∈   Ω×{0}
    t̂ = 0.0

    x₀ =0.0                 # random (x₀,t₀)∈ {0}×[0,T]
    t₀ = rand(0:0.01:3.0)

    ∇u = Flux.gradient(u.π) do
        cost(x, t,  x̂, t̂, x₀, t₀)
    end

    Adam_update(u, ∇u)
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
