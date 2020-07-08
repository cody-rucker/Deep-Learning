
struct Layer
    W
    b
end

Layer(in::Integer, out::Integer) = Layer(randn(out, in), randn(out))

#(m::Layer)(x) = m.W * x .+ m.b

#a = Layer(10, 5)

#a(rand(10)) # => 5-element vector

θ = Dict(:W1 => rand(5,2), :b1 => rand(5,1),
         :W2 => rand(1,5), :b2 => rand(1,1))
Layer(x, θ) = θ[:W] * x .+ θ[:b]

u(x, θ) =θ[:W2] * σ.(θ[:W1] *x .+ θ[:b1]) .+ θ[:b2]

#u(x, θ) = layer(σ.(layer(x, θ[1])), θ[2])
