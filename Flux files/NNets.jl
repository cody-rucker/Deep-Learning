using Flux
using ForwardDiff
using Zygote

# create type to hold adaptive moment estimation parameters
"""
    Adam(N, M)
Defines a composite type to store a neural networks' vital parameters.

In addition to storing weight or bias information, this type also stores necesary
parameters for adaptive moment estimation.

# Arguments
- `N::Integer`: the desired input dimension.
- `M::Integer`: the desired output dimension.

# Examples
```jldoctest
julia> W = Adam(5, 1)
```
"""
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

  function Adam(N::Int64, M::Int64)
      θ = rand(N, M)
      ∇ = rand(N, M)
      m = zeros(size(θ))
      v = zeros(size(θ))
      β₁= 0.9
      β₂= 0.999
      α = 0.01
      ϵ = 1e-8
      t = 0
    new(θ, ∇, m, v, β₁, β₂, α, ϵ, t)
  end

end


"""

    NeuralNet2(hidden_dim, out_dim)

Defines a trainable neural network with `hidden_dim` neurons in each layer and
`out_dim` entries in its output.

Weight and Bias parameters are stored as the composite Adam type to prepare the
NN for adaptive moment estimation while training.

# Arguments
- `hidden_dim::Integer`: the desired hidden dimension.
- `out_dim::Integer`: the desired output dimension.

# Examples
```jldoctest
julia> u = NeuralNet2(5,1)
julia> u(1.0, 2.0)
1×1 Array{Float64,2}:
 3.338606039920986
```
"""
mutable struct NeuralNet2
    Wₓ::Adam
    Wᵧ::Adam
    b₁::Adam
    W₂::Adam
    b₂::Adam
    W₃::Adam
    b₃::Adam

    Π::Tuple
    π::Flux.Params

    function NeuralNet2(hidden_dims::Int64, out_dims::Int64, spatial_dim::Int64=2)
        Wₓ = Adam(hidden_dims, 1)
        Wᵧ = Adam(hidden_dims, 1)
        b₁ = Adam(hidden_dims, 1)
        W₂ = Adam(hidden_dims, hidden_dims)
        b₂ = Adam(hidden_dims, 1)
        W₃ = Adam(out_dims, hidden_dims)
        b₃ = Adam(out_dims, 1)


        Π = (Wₓ, Wᵧ, b₁, W₂, b₂, W₃, b₃)
        π = Flux.params(Wₓ.θ, Wᵧ.θ, b₁.θ, W₂.θ, b₂.θ, W₃.θ, b₃.θ)
        new(Wₓ, Wᵧ, b₁, W₂, b₂, W₃, b₃, Π, π);
    end
end

(u::NeuralNet2)(x, y) = u.W₃.θ * σ.(
                        u.W₂.θ * σ.(
                        u.Wₓ.θ*x + u.Wᵧ.θ*y .+ u.b₁.θ) .+ u.b₂.θ) .+ u.b₃.θ ;


"""

    FirstNetDerivative(u, "x")

Defines a neural network which inherits weight and bias parameters from `u` and
outputs the derivative of `u` with respect to `"x"`.

# Arguments
- `u::NeuralNet2`: neural net to be differentiated.
- `"x"::String`: which variable we are differentiating with respect to. "x" and "y" are both viable inputs.

# Examples
```jldoctest
julia> u = NeuralNet2(5,1)
julia> uₓ = FirstNetDerivative(u, "x")
julia> uₓ(1.0, 2.0)
1×1 Array{Float64,2}:
 0.027888484050549646
```
"""
mutable struct FirstNetDerivative
    Wₓ::Adam
    Wᵧ::Adam
    b₁::Adam
    W₂::Adam
    b₂::Adam
    W₃::Adam
    b₃::Adam

    dξ::AbstractArray

    function FirstNetDerivative(u::NeuralNet2, differential)
        Wₓ = u.Wₓ
        Wᵧ = u.Wᵧ
        b₁ = u.b₁
        W₂ = u.W₂
        b₂ = u.b₂
        W₃ = u.W₃
        b₃ = u.b₃

        if differential == "x"
            dξ = Wₓ.θ

        elseif differential == "y"
            dξ = Wᵧ.θ

        else
            print("Must specify x or y as a string literal argument.")
        end

        new(Wₓ, Wᵧ, b₁, W₂, b₂, W₃, b₃, dξ)
    end


end

(u::FirstNetDerivative)(x, y) = u.W₃.θ * (σ'.(u.W₂.θ * σ.( u.Wₓ.θ*x .+ u.Wᵧ.θ*y .+ u.b₁.θ) .+ u.b₂.θ) .*
                  (u.W₂.θ * (σ'.( u.Wₓ.θ*x .+ u.Wᵧ.θ*y .+ u.b₁.θ) .* u.dξ )))


"""

  SecondNetDerivative(u, "x", "y")

Defines a neural network which inherits weight and bias parameters from `u` and
outputs the second derivative of `u`. `"x"` and `"y"` are valid inputs allowing
you to specify any second-order derivative of `u`.

# Arguments
- `u::NeuralNet2`: neural net to be differentiated.
- `"x"::String`: specifies a derivative w.r.t first input variable.
- `"y"::String`: specifies a derivative w.r.t second input variable.

# Examples
```jldoctest
julia> u = NeuralNet2(5,1)
julia> uₓᵧ = SecondNetDerivative(u, "x", "y"))
julia> uₓᵧ(1.0, 2.0)
1×1 Array{Float64,2}:
 -0.012153494673076519
```
"""
mutable struct SecondNetDerivative
  Wₓ::Adam
  Wᵧ::Adam
  b₁::Adam
  W₂::Adam
  b₂::Adam
  W₃::Adam
  b₃::Adam

  dξ::AbstractArray
  dζ::AbstractArray

  function SecondNetDerivative(u::NeuralNet2, d₁, d₂)
    Wₓ = u.Wₓ
    Wᵧ = u.Wᵧ
    b₁ = u.b₁
    W₂ = u.W₂
    b₂ = u.b₂
    W₃ = u.W₃
    b₃ = u.b₃

    if d₁ == "x"
        dξ = Wₓ.θ

    elseif d₁ == "y"
        dξ = Wᵧ.θ

    else
        print("Must specify x or y as a string literal argument.")
    end

    if d₂ == "x"
        dζ = Wₓ.θ

    elseif d₂ == "y"
        dζ = Wᵧ.θ

    else
        print("Must specify x or y as a string literal argument.")
    end

    new(Wₓ, Wᵧ, b₁, W₂, b₂, W₃, b₃, dξ, dζ)
  end
end

function (u::SecondNetDerivative)(x, y)
    Σ = u.Wₓ.θ*x .+ u.Wᵧ.θ*y .+ u.b₁.θ

    a = σ''.(u.W₂.θ * σ.(Σ) .+ u.b₂.θ) .* (u.W₂.θ * (σ'.(Σ) .* u.dζ)) .* (u.W₂.θ * (σ'.(Σ) .* u.dξ))

    b = σ'.( u.W₂.θ * σ.(Σ) .+ u.b₂.θ) .* (u.W₂.θ * (σ''.(Σ) .* u.dξ .* u.dζ) )

    return u.W₃.θ * (a .+ b)
end

# given a parameter and the gradient w.r.t that parameter,
# perform an adaptive moment update
function Adam_step(P::Adam, ∇P::AbstractArray)
    P.∇ = ∇P
    P.t += 1

    P.m = P.β₁ * P.m + (1 - P.β₁) .* P.∇
    P.v = P.β₂ * P.v + (1 - P.β₂) .* (P.∇).^2
    m̂ = P.m / (1 - (P.β₁).^(P.t))
    v̂ = P.v / (1 - (P.β₂).^(P.t))

    # update parameter
    P.θ .-= P.α .* m̂ ./ (sqrt.(v̂) .+ P.ϵ)
end

# given a NN and its gradient, update each parameter
function Adam_update(u::NeuralNet2, ∇u)
    for p in u.Π
        Adam_step(p, ∇u[p.θ])
    end
end








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
