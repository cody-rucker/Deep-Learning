using ChainRules
struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(randn(out, in), randn(out))

#(m::Affine)(x, t) = m.W[:, 1] * x + m.W[:, 2]*t .+ m.b
(m::Affine)(x) = m.W*x .+ m.b
a = Affine(1, 1)
function u(x)
    y₁ = a.W * x
    y₂ = y₁ .+ a.b
    y₃ = broadcast(tanh, y₂)
end

function ChainRulesCore.frule((Δself, ẋ), ::typeof(a), x)
    Y = a(x)
    function a_pushforward(Δself, ẋ)
        return a.W *ẋ
    end
    return Y, a_pushforward(Δself, ẋ)
end

# calculate forward propogation of input wiggles
y₁, y₁̇ = frule((NO_FIELDS, 1), a, x)
y₂, y₂̇ = frule((NO_FIELDS, y₁̇  ), +, y₁, a.b)
#=
function ChainRules.frule((NO_FIELDS, Δx, Δt ), ::typeof(a), x, t)
  function pushforward(Δx, Δt)
    return a.W[:,1]*Δx + a.W[:, 2]*Δt #ChainRules.NO_FIELDS, a.W[:, 1]*dargx, a.W[:, 2]*dargt
   end
   return a(x, t), pushforward
 end
=#
 # good:
 #function frule(::typeof(foo), x)
#     Y = foo(x)
#     function foo_pushforward(_, ẋ)
#         return bar(ẋ)
#     end
#     return Y, foo_pushforward
 #end

 # good:
function ChainRulesCore.frule((Δself, ẋ, ṫ), ::typeof(a), x, t)
    Y = a(x, t)
    function a_pushforward(Δself, ẋ, ṫ)
        return a.W[:, 1]ẋ + a.W[:, 2]ṫ
    end
    return Y, a_pushforward(Δself, ẋ, ṫ)
end

function foo(x, t)
    q = tanh.(a(x, t))
end
#=
function rrule(::typeof(foo), x)
    Y = foo(x)
    function foo_pullback(x̄)
        return NO_FIELDS, bar(x̄)
    end
    return Y, foo_pullback
end
=#
#=
function foo(x)
    u = sin(x)
    v = asin(b)
    return v
end

x = π/4
ẋ = 1 #∂x/∂x

u, u̇ = frule((NO_FIELDS, ẋ), sin, x) # ∂u/∂x
# frule takes arguments
#   frule((Δf, Δx...), f, x...)
# so in the following example * has nofield and 2 is constant so
# our first two args below are dself and the final one is the derivative of
# a w.r.t x
v, v̇ = frule((NO_FIELDS, u̇), asin, u) # ∂v/∂x = ∂v/∂u * ∂u/∂x

# Try doing this to a NN model

function û(x, W, b)
    z = W*x
    z₁ = z .+ b
    u = tanh.(z₁)
    return u
end
=#
