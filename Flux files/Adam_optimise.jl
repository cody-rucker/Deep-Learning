

mutable struct Adam
  θ::AbstractArray{Float32} # Parameter array
  ∇::AbstractArray{Float32}                # Gradient function
  m::AbstractArray{Float32}     # First moment
  v::AbstractArray{Float32}     # Second moment
  β₁::Float64                   # Exp. decay first moment
  β₂::Float64                   # Exp. decay second moment
  α::Float64                    # Step size
  ϵ::Float64                  # Epsilon for stability
  t::Int                        # Time step (iteration)
end

function Adam(theta::AbstractArray{Float32}, ∇::AbstractArray{Float32})
  m   = zeros(size(theta))
  v   = zeros(size(theta))
  β₁  = 0.9
  β₂  = 0.999
  α   = 0.01
  ϵ   = 1e-8
  t   = 0
  Adam(theta, ∇, m, v, β₁, β₂, α, ϵ, t)
end

function Adam_update!(W::AbstractArray{Float32}, W̄::Adam, ∇u::AbstractArray{Float32}, t::Int64)
    W̄.∇ = ∇u
    W̄.m = W̄.β₁ * W̄.m + (1 - W̄.β₁) .* W̄.∇
    W̄.v = W̄.β₂ * W̄.v + (1 - W̄.β₂) .* (W̄.∇).^2
    m̂ = W̄.m / (1 - (W̄.β₁).^t)
    v̂ = W̄.v / (1 - (W̄.β₂).^t)

    W .-= W̄.α .* m̂ ./ (sqrt.(v̂) .+ W̄.ϵ)

end
