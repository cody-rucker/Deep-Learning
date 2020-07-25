using Flux
using ForwardDiff
using Zygote
using ChainRulesCore
using ProgressMeter
using Plots
using LinearAlgebra
include("Adam_optimise.jl")

struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) = Affine(randn(out, in), randn(out))

struct InputLayer
    Wₓ
    Wₜ
    b
end

InputLayer(in::Integer, out::Integer) = InputLayer(randn(out, 1), randn(out, 1), randn(out))

(m::InputLayer)(x, t) = m.Wₓ*x .+ m.Wₜ*t .+ m.b

(m::Affine)(x) = m.W*x .+ m.b

# ChainRules
function ChainRulesCore.frule((Δself, ẋ, ṫ), β::InputLayer, x, t)
    Y = β(x, t)
    function pushforward(Δself, ẋ, ṫ)
        return β.Wₓ *ẋ + β.Wₜ*ṫ
    end
    return Y, pushforward(Δself, ẋ, ṫ)
end

function ChainRulesCore.frule((Δself, ẋ), α::Affine, x)
    Y = α(x)
    function pushforward(Δself, ẋ)
        return α.W * ẋ
    end
    return Y, pushforward(Δself, ẋ)
end

function ChainRulesCore.frule((Δself, ẋ), ::typeof(σ), x)
    Y = σ.(x)
    function pushforward(Δself, ẋ)
        return σ'.(x).*ẋ
    end
    return Y, pushforward(Δself, ẋ)
end

function ChainRulesCore.frule((Δself, ẋ), ::typeof(σ'), x)
    Y = σ'.(x)
    function pushforward(Δself, ẋ)
        return σ''.(x).*ẋ
    end
    return Y, pushforward(Δself, ẋ)
end

function ChainRulesCore.frule((Δself, ẋ, ẏ), ::typeof(*), x, y)
    Y = x .* y
    function pushforward(Δself, ẋ, ẏ)
        return y.*ẋ + x.*ẏ
    end
    return Y, pushforward(Δself, ẋ, ẏ)
end

# need an extra multiplication operator
function φ(x, y)
    return y*x
end

function ChainRulesCore.frule((Δself, ẋ, ẏ), ::typeof(φ), x, y)
    Y = φ(x, y)
    function pushforward(Δself, ẋ, ẏ)
        return y*ẋ
    end
    return Y, pushforward(Δself, ẋ, ẏ)
end


M = 10
batch_size = 32

# initial conditions on displacement(η) and velocity (γ)
η(x) = sin(π*x)
γ(x) = -2π*cos(2π * x)

# boundary condition at x = 0 and x = 1
g₀(t) = 0.0
g₁(t) = 0.0

# exact solution
uexact(x,t) = exp(-2*t)*sin.(pi*x)

# correction data to satisfy uₜ - uₓₓ
F(x, t) = (π^2 - 2) * exp(-2*t) * sin(π*x)

######################################################

v = InputLayer(2,20)
z = Affine(20,1)
θ = Flux.params(v.Wₓ, v.Wₜ, v.b, z.W, z.b)

function u(x, t)
    u1 = v(x, t)
    u2 = σ.(u1)
    u3 = z(u2)
    return sum(u3)
end

function uₓ(x, t)
    u1 = v(x, t)
    u2 = σ'.(u1)
    u3 = u2 .* v.Wₓ
    u4 = z.W * u3
    return sum(u4)
end

function uₜ(x, t)
    u1 = v(x, t)
    u2 = σ'.(u1)
    u3 = u2 .* v.Wₜ
    u4 = z.W * u3
    return sum(u4)
end

dσ(x) =  σ(x) * (1 - σ(x))
d²σ(x) = dσ(x) - 2* dσ(x)*σ(x)

#=
function uₓₓ(x, t)
    Σ = v.Wₓ*x .+ v.Wₜ*t .+ v.b

    a = d²σ.(z.W * σ.(Σ) .+ z.b) .* (z.W * (dσ.(Σ) .* v.Wₓ)) .* (z.W * (dσ.(Σ) .* v.Wₓ))

    b = dσ.( z.W * σ.(Σ) .+ z.b) .* (z.W * (d²σ.(Σ) .* v.Wₓ .* v.Wₓ) )

    return sum((a .+ b))
end
=#


function uₓₓ(x, t)
    u1, u̇1 = frule((NO_FIELDS, 1, Zero()), v, x, t)
    u2, u̇2 = frule((NO_FIELDS, u̇1), σ', u1)
    u3, u̇3 = frule((NO_FIELDS, u̇2, Zero()), *, u2, v.Wₓ)
    u4, u̇4 = frule((NO_FIELDS, u̇3, Zero()), φ, u3, z.W)

    return sum(u̇4)
end

function uₓ(x, t)
    u1, u̇1 = frule((NO_FIELDS, 1, 0), v, x, t)
    u2, u̇2 = frule((NO_FIELDS, unthunk(u̇1)), σ, u1)
    u3, u̇3 = frule((NO_FIELDS, unthunk(u̇2)), z, u2)
    return sum(u̇3)
end

ux = (x, t)->ForwardDiff.derivative(x->u(x, t), x)

uxx = (x, t)-> ForwardDiff.derivative(x->ux(x, t), x)


#@assert uₓ(x, t) ≈ ux(x, t)

#uₓₓ = (x, t) -> ForwardDiff.derivative(x -> uₓ(x, t), x)

#=
function uₜₜ(x, t)
    Σ = Wₓ*x .+ Wₜ*t .+ b1

    a = d²σ.(W2 * σ.(Σ) .+ b2) .* (W2 * (dσ.(Σ) .* Wₜ)) .* (W2 * (dσ.(Σ) .* Wₜ))

    b = dσ.( W2 * σ.(Σ) .+ b2) .* (W2 * (d²σ.(Σ) .* Wₜ .* Wₜ) )

    return sum(W3 * (a .+ b))
end
=#
#@assert uₜₜ(x, t) ≈ ForwardDiff.derivative(t -> ut(x, t), t)

#=
Define a physics informed neural net
            f := uₜₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
function f(x, t)
    return uₜ(x, t)- uₓₓ(x, t)
end

# cost function to be minimized during gradient descent
function cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
    sum(abs.(f(x,  t) .- F(x, t)).^2 +                  # enforce structure of the PDE
        abs.(u(x̂,  t̂) .- η.(x̂)).^2 +                    # initial displacement
        abs.(u(x₀, t₀) .- g₀.(t₀)).^2 +                 # b.c at x=0
        abs.(u(x₁, t₁) .- g₁.(t₁)).^2   )               # b.c at x=1
end

# initialize Adam objects to store optimization parameters
#Wx = Adam(Wₓ, Wₓ)
#Wt = Adam(Wₜ, Wₜ)
#b₁ = Adam(b1, b1)

#W₂ = Adam(W2, W2)
#b₂ = Adam(b2, b2)

#W₃ = Adam(W3, W3)
#b₃ = Adam(b3, b3)
Wx = Adam(v.Wₓ, v.Wₓ)
Wt = Adam(v.Wₜ, v.Wₜ)
b₁ = Adam(v.b, v.b)

W₂ = Adam(z.W, z.W)
b₂ = Adam(z.b, z.b)

#W₃ = Adam(W3, W3)
#b₃ = Adam(b3, b3)

# training loop: Adam optimization
@showprogress "Training..." for n = 1:M
    sleep(0.1)
    for i = 1:batch_size



        x = rand(0:0.001:1, 1, 1)[1]              # random x∈   Ω×[0,T]
        t = rand(0:0.001:1, 1, 1)[1]
        x̂ = rand(0:0.001:1, 1, 1)[1]      # random x∈   Ω×{0}
        t̂ = 0.0
        ẋ = rand(0:0.001:1, 1, 1)[1]      # random x∈   Ω×{0}
        ṫ = 0.0
        x₀ =0.0                         # random x∈ {0}×[0,T]
        t₀ = rand(0:0.001:1.0, 1, 1)[1]
        x₁ = 1.0                         # random x∈ {1}×[0,T]
        t₁ = rand(0:0.001:1.0, 1, 1)[1]

        ∇u = gradient(θ) do
            cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
        end

        # Adam optimisation
        Adam_update!(v.Wₓ, Wx, ∇u[v.Wₓ], i)
        Adam_update!(v.Wₜ, Wt, ∇u[v.Wₜ], i)
        Adam_update!(v.b, b₁, ∇u[v.b], i)
        Adam_update!(z.W, W₂, ∇u[z.W], i)
        Adam_update!(z.b, b₂, ∇u[z.b], i)
        #Adam_update!(W3, W₃, ∇u[W3], i)
        #Adam_update!(b3, b₃, ∇u[b3], i)
    #    Adam_update!(θ[:W3], W₃, ∇u[:W3], i)
    #    Adam_update!(θ[:b3], b₃, ∇u[:b3], i)

    # Stochastic gradient descent (SGD)
    #    @inbounds for j = 1:length(θ)
    #        θ[j] .-= α .* ∇u[θ[j]]
    #    end
    #    @show cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
    end
end

T = 0:0.001:1
xfine = 0:0.001:1
Z = zeros(length(xfine), length(T))

uexact(x, t) = exp(-2t) * sin(π.*(x))
err(x, t) = uexact.(x, t) - u(x, t)# θ)
#plot(legend=true, size=(500,500), xlim=(0,1), ylim=(-1.2,1.2))

for i = 1:length(T)
    for j = 1:length(xfine)
        Z[j,i] = u.(xfine[j], T[i])
    end
end
#=
@inbounds for i = 1:2:length(T)
    p1 = plot(xfine, u.(xfine[:], T[i]), size=(1000, 750), ylims=(0, 1.0), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], T[i]), label="exact")

    p2 = plot(xfine, err.(xfine[:], T[i]), ylims = (-0.1, 0.1), label="error", color=:red)

    p = plot(p1, p2, layout = (2,1), size=(1000, 750))

    #plot!(xfine, uexact.(xfine[:] .+ c*t[i]), label="exact")
    display(p)
end
=#
