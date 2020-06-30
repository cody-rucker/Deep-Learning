
struct Affine
    W
    b
end

Affine(in::Integer, out::Integer) = Affine(randn(out, in), randn(out))

(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10, 5)

a(rand(10)) # => 5-element vector
