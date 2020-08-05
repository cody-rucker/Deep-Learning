using Flux


mutable struct Adam
  θ::AbstractArray{Float32}     # Parameter array
  ∇::AbstractArray{Float32}     # Gradient w.r.t θ
  m::AbstractArray{Float32}     # First moment
  v::AbstractArray{Float32}     # Second moment
  β₁::Float64                   # Exp. decay first moment
  β₂::Float64                   # Exp. decay second moment
  α::Float64                    # Step size
  ϵ::Float64                    # Epsilon for stability
  t::Int                        # Time step (iteration)
end

"""
    Adam()
Locate the element of maximal absolute value in the one-dimensional array a.
# Examples
```jldoctest
julia> A = [ 1 2 -3]
julia> j = find_pivot(A)
```
"""
function Adam(θ::AbstractArray{Float64}, ∇::AbstractArray{Float64})
  m   = zeros(size(θ))
  v   = zeros(size(θ))
  β₁  = 0.9
  β₂  = 0.999
  α   = 0.01
  ϵ   = 1e-8
  t   = 0
  Adam(θ, ∇, m, v, β₁, β₂, α, ϵ, t)
end

function Adam_update!(θ::AbstractArray{Float64}, W̄::Adam, ∇θ::AbstractArray{Float64}, t::Int64)
    # initialize parameter array along with 1st and second moments
    W̄.∇ = ∇θ

    W̄.m = W̄.β₁ * W̄.m + (1 - W̄.β₁) .* W̄.∇
    W̄.v = W̄.β₂ * W̄.v + (1 - W̄.β₂) .* (W̄.∇).^2
    m̂ = W̄.m / (1 - (W̄.β₁).^t)
    v̂ = W̄.v / (1 - (W̄.β₂).^t)

    # update parameter
    θ .-= W̄.α .* m̂ ./ (sqrt.(v̂) .+ W̄.ϵ)

    return θ
end
