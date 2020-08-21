using Flux
using ForwardDiff

mutable struct NeuralNet2
    Wₓ::AbstractArray
    Wᵧ::AbstractArray
    b₁::AbstractArray
    W₂::AbstractArray
    b₂::AbstractArray
    W₃::AbstractArray
    b₃::AbstractArray

    θ::Flux.Params
end

function NeuralNet2(hidden_dims::Int64, out_dims::Int64)
    Wₓ = rand(hidden_dims, 1)
    Wᵧ = rand(hidden_dims, 1)
    b₁ = rand(hidden_dims, 1)
    W₂ = rand(hidden_dims, hidden_dims)
    b₂ = rand(hidden_dims)
    W₃ = rand(out_dims, hidden_dims)
    b₃ = rand(out_dims)

    θ = Flux.params(Wₓ, Wᵧ, b₁, W₂, b₂, W₃, b₃)
    NeuralNet2(Wₓ, Wᵧ, b₁, W₂, b₂, W₃, b₃, θ);
end

(u::NeuralNet2)(x, y) = u.W₃ * σ.(
                        u.W₂ * σ.(
                        u.Wₓ*x + u.Wᵧ*y .+ u.b₁) .+ u.b₂) .+ u.b₃ ;


mutable struct NeuralNet3
    Wₓ::AbstractArray
    Wᵧ::AbstractArray
    Wₜ::AbstractArray
    b₁::AbstractArray
    W₂::AbstractArray
    b₂::AbstractArray
    W₃::AbstractArray
    b₃::AbstractArray

    θ::Flux.Params

end

function NeuralNet3(hidden_dims::Int64, out_dims::Int64)
    Wₓ = rand(hidden_dims, 1)
    Wᵧ = rand(hidden_dims, 1)
    Wₜ = rand(hidden_dims, 1)
    b₁ = rand(hidden_dims, 1)
    W₂ = rand(hidden_dims, hidden_dims)
    b₂ = rand(hidden_dims)
    W₃ = rand(out_dims, hidden_dims)
    b₃ = rand(out_dims)

    θ = Flux.params(Wₓ, Wᵧ, Wₜ, b₁, W₂, b₂, W₃, b₃)


    NeuralNet3(Wₓ, Wᵧ, Wₜ, b₁, W₂, b₂, W₃, b₃, θ)
end

(u::NeuralNet3)(x, y, t) = u.W₃ * σ.(
                          u.W₂ * σ.(
                          u.Wₓ*x + u.Wᵧ*y + u.Wₜ*t .+ u.b₁) .+ u.b₂) .+ u.b₃


# create type to hold adaptive moment estimation parameters

mutable struct Adam
  θ::AbstractArray     # Parameter array
  ∇::AbstractArray     # Gradient w.r.t θ
  m::AbstractArray     # First moment
  v::AbstractArray     # Second moment
  β₁::Float64                   # Exp. decay first moment
  β₂::Float64                   # Exp. decay second moment
  α::Float64                    # Step size
  ϵ::Float64                    # Epsilon for stability
  t::Int                        # Time step (iteration)
end

function Adam(θ::AbstractArray)
    ∇  = zeros(size(θ))
    m  = zeros(size(θ))
    v  = zeros(size(θ))
    β₁ = 0.9
    β₂ = 0.999
    α  = 0.01
    ϵ  = 1e-8
    t  = 0
  Adam(θ, ∇, m, v, β₁, β₂, α, ϵ, t)
end

# given a neural network, return adaptive moment parameters for
# each weight and bias

function learning_parameters(u::NeuralNet2)
    W̄ₓ = Adam(u.Wₓ)
    W̄ᵧ = Adam(u.Wᵧ)
    W̄ₜ = Adam(u.Wₜ)
    b̄₁ = Adam(u.b₁)
    W̄₂ = Adam(u.W₂)
    b̄₂ = Adam(u.b₂)
    W̄₃ = Adam(u.W₃)
    b̄₃ = Adam(u.b₃)

    return W̄ₓ, W̄ᵧ, W̄ₜ, b̄₁, W̄₂, b̄₂, W̄₃, b̄₃
end
