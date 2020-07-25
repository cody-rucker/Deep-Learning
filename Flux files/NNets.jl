mutable struct Affine
  W
  b
  f
end

Affine(in::Integer, out::Integer, f::Function=identity) = Affine(randn(out, in), randn(out), f)

mutable struct InputLayer
    Wₓ
    Wₜ
    b
    f
end

InputLayer(in::Integer, out::Integer, f::Function=identity) = InputLayer(randn(out, 1), randn(out, 1), randn(out), f)

(m::InputLayer)(x, t) =m.f( m.Wₓ*x .+ m.Wₜ*t .+ m.b)

(m::Affine)(x) = m.f(m.W*x .+ m.b)

struct NeuralNet
    layers
    f
end

function NeuralNet(a::InputLayer, b::Affine...)
    x = vcat(a, b[1])
    for i = 2:length(b)
        x = vcat(x, b[i])
    end

    f = b[1]∘a
    for i = 2:length(b)
        f = b[i]∘f
    end
        NeuralNet(x, f)
end
(m::NeuralNet)(x, t) = sum(m.f(x, t))
