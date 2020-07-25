using ChainRules
using Flux
using Zygote
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
#(m::Affine)(x, t) = m.W[:, 1] * x + m.W[:, 2]*t .+ m.b
(m::Affine)(x) = m.W*x .+ m.b

# ChainRules
function frule((Δself, ẋ, ṫ), β::InputLayer, x, t)
    Y = β(x, t)
    function β_pushforward(Δself, ẋ, ṫ)
        return β.Wₓ *ẋ + β.Wₜ*ṫ
    end
    return Y, β_pushforward
end

function frule((Δself, ẋ), α::Affine, x)
    Y = α(x)
    function α_pushforward(Δself, ẋ)
        return α.W * ẋ
    end
    return Y, α_pushforward
end

function frule((Δself, ẋ), ::typeof(σ), x)
    Y = σ.(x)
    function σ_pushforward(Δself, ẋ)
        return σ'.(x).*ẋ
    end
    return Y, σ_pushforward
end


function aₓ(x, t)
    z1, ż1 = frule((θ, 1, 0), a, x, t)
    z2, ż2 = frule((θ, unthunk(ż1)), σ, z1)
    z3, ż3 = frule((θ, unthunk(ż2)), b, z2)

    return sum(ż3(θ, ż2(θ, ż1(θ, 1, 0))))
end

a = InputLayer(2,3)
b = Affine(3,1)
θ = params(a.Wₓ, a.Wₜ, a.b, b.W, b.b)

function g(x,t)
    z1 = a(x,t)
    z2 = σ.(z1)
    z3 = b(z2)
end

z1, ż1 = frule((θ, 1, 0), a, x, t)
z2, ż2 =  frule((θ, unthunk(ż1)), σ, z1)
z3, ż3 = frule((θ, unthunk(ż2)), b, z2)

aₓₓ = (x, t)-> Zygote.forwarddiff(x->aₓ(x, t), x)

∇u = gradient(θ) do
       aₓₓ(x, t)
       end
