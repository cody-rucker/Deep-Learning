# learning about dual numbers in automatic differentiation
struct Dual{T<:Real} <: Real
    x::T
    ϵ::T
end

# nice representation for this
Base.show(io::IO, d::Dual) = print(io, d.x, " + ", d.ϵ, "ϵ")

Dual(1, 2)

# add some of our differentiation rules
import Base: +, -, *, /
a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)
a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x * b.x, b.x * a.ϵ - a.x * b.ϵ)

Base.sin(d::Dual) = Dual(sin(d.x), d.ϵ * cos(d.x))
Base.cos(d::Dual) = Dual(cos(d.x), - d.ϵ * sin(d.x))

Dual(2, 2) * Dual(3, 4)

Base.convert(::Type{Dual{T}}, x::Dual) where T = Dual(convert(T, x.x), convert(T, x.ϵ))
Base.convert(::Type{Dual{T}}, x::Real) where T = Dual(convert(T, x), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}

Dual(1, 2) * 3

# we can make a utility which allows us to differentiate any function
D(f, x) = f(Dual(x, one(x))).ϵ



# implementing tensorflow
include("utils.jl")

import Base: +, -

struct Staged
  w::Wengert
  var
end

a::Staged + b::Staged = Staged(w, push!(a.w, :($(a.var) + $(b.var))))

a::Staged - b::Staged = Staged(w, push!(a.w, :($(a.var) - $(b.var))))

for f in [:+, :*, :-, :^, :/]
  @eval Base.$f(a::Staged, b::Staged) = Staged(a.w, push!(a.w, Expr(:call, $(Expr(:quote, f)), a.var, b.var)))
  @eval Base.$f(a, b::Staged) = Staged(b.w, push!(b.w, Expr(:call, $(Expr(:quote, f)), a, b.var)))
end

w = Wengert()
x = Staged(w, :x)
y = Staged(w, :y)
