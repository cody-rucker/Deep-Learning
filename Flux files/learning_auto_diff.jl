
# derice an expression for the derivative
# of an expression ex
function derive(ex, x)
    # if ex == x return 1
    ex == x ? 1 :
    # if ex is a number or symbol, return 0
    ex isa Union{Number, Symbol} ? 0 :
    error("$ex in not differentiable")
end

using MacroTools

y = :(x + 1)
# if @capture returns true, we can work with the subexpressions
# a and b

@capture(y, a_ * b_)    # false

@capture(y, a_ + b_)    # true

# use this to add a rule to the chain rule

function derive(ex, x)
    ex == x ? 1 :
    ex isa Union{Number, Symbol} ? 0 :
    @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
    error("$ex is not differentiable")
end

y = :(1 + x)
derive(y, :x) # returns :(1 + 0)

# build in the rest of the derivative rules similarly
function derive(ex, x)
    ex == x ? 1 :
    ex isa Union{Number, Symbol} ? 0 :
    # sum rule
    @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
    # product rule
    @capture(ex, a_ * b_) ? :($a * $(derive(b, x)) + $b * $(derive(a, x))) :
    # power rule
    @capture(ex, a_^n_Number) ? :($(derive(a, x)) * ($n * $a^$(n-1))) :
    # quotient rule
    @capture(ex, a_ / b_) ? :( $b * $(derive(a, x)) - $a * $(derive(b, x)) / $b^2) :
    error("$ex is not differentiable")
end

# test on
y = :(3x^2 + (2x + 1))
dy = derive(y, :x)

# clean this up by introducing smarter functions that avoid being redundant
addm(a, b) = a == 0 ? b : b == 0 ? a : :($a + $b)
mulm(a, b) = 0 in (a, b) ? 0 : a == 1 ? b : b == 1 ? a : :($a * $b)
mulm(a, b, c...) = mulm(mulm(a, b), c...)

# tweak the derive function to utilize these clean up functions
function derive(ex, x)
    ex == x ? 1 :
    ex isa Union{Number, Symbol} ? 0 :
    # sum rule
    @capture(ex, a_ + b_) ? addm(derive(a, x) , derive(b, x)) :
    # product rule
    @capture(ex, a_ * b_) ? addm(mulm(a, derive(b, x)) , mulm(b, derive(a, x))) :
    # power rule
    @capture(ex, a_^n_Number) ? mulm(derive(a, x), n, :($a^$(n-1))) :
    # quotient rule
    @capture(ex, a_ / b_) ? :($(mulm(b, derive(a, x))) - $(mulm(a, derive(b, x))) / $b^2) :
    error("$ex is not differentiable")
end

# test on and see the output is much cleaner
y = :(3x^2 + (2x + 1))
dy = derive(y, :x)

# moreover you can call this multiple times
ddy = derive(dy, :x)

# however because of the way the derivative rules are expanded out, if we
# call
derive(:(x / (1 + x^2)), :x)
derive(:(x / (1 + x^2) * x), :x)

# the only difference is an extra *x but the result is an output which is
# exponentially large in the size of its input. This expression is not actually
# as big as it looks. Imagine
y1 = :(1 *2)
y2 = :($y1 + $y1 + $y1 + $y1 )
#out: :(1 * 2 + 1 * 2 + 1 * 2 + 1 * 2)
# though it looks large, this is actually just four pointers to y1 and the
# resulting graph is printed as a tree. This can be seen by printing
# the epxression in a way that preserves the structure

printstructure(x, _, _) = x

function printstructure(ex::Expr, cache = IdDict(), n = Ref(0))
    haskey(cache, ex) && return cache[ex]
    args = map(x -> printstructure(x, cache, n), ex.args)
    cache[ex] = sym = Symbol(:y, n[] += 1)
    println(:($sym = $(Expr(ex.head, args...))))
    return sym
end

#Note that this is not the same as running common subexpression elimination to
#simplify the tree, which would have an $O(n^2)$ computational cost. If there is
#real duplication in the expression, it'll show up.

#=
derive(:(x / (1 + x^2)), :x) |> printstructure;
out:
y1 = x ^ 2
y2 = 1 + y1
y3 = x ^ 1
y4 = 2y3
y5 = x * y4
y6 = y2 ^ 2
y7 = y5 / y6
y8 = y2 - y7
=#

#= now the expression x² +1 is defined once and reused rather than being repeated
addinf the extra *x leads to a few more instructions but no longer doubles the
output size
=#

include("utils.jl")
# we can port our recursive symbolic differentiation algorithm to the
# Wengert list

function derive(ex, x, w)
    ex isa Variable && (ex = w[ex])
    ex == x ? 1 :
    ex isa Union{Number, Symbol} ? 0 :
    # sum rule
    @capture(ex, a_ + b_) ? push!(w, addm(derive(a, x, w) , derive(b, x, w))) :
    # product rule
    @capture(ex, a_ * b_) ? push!(w, addm(mulm(a, derive(b, x, w)) , mulm(b, derive(a, x, w)))) :
    # power rule
    @capture(ex, a_^n_Number) ? push!(w, mulm(derive(a, x, w), n, :($a^$(n-1)))) :
    # quotient rule
    @capture(ex, a_ / b_) ? push!(w, :($(mulm(b, derive(a, x, w))) - $(mulm(a, derive(b, x, w))) / $b^2)) :
    error("$ex is not differentiable")
end

# this differentiation algorithm begins with dx/dx = 1 and propogates this
# forward to the output. i.e this is Forward mode differentiation
function derive(w::Wengert, x)
  ds = Dict()
  ds[x] = 1
  d(x) = get(ds, x, 0)
  for v in keys(w)
    ex = w[v]
    Δ = @capture(ex, a_ + b_) ? addm(d(a), d(b)) :
        @capture(ex, a_ * b_) ? addm(mulm(a, d(b)), mulm(b, d(a))) :
        @capture(ex, a_^n_Number) ? mulm(d(a),n,:($a^$(n-1))) :
        @capture(ex, a_ / b_) ? :($(mulm(b, d(a))) - $(mulm(a, d(b))) / $b^2) :
        error("$ex is not differentiable")
    ds[v] = push!(w, Δ)
  end
  return w
end

derive(Wengert(:(x / (1 + x^2))), :x) |> Expr

# with a few tweaks we can get reverse mode differentiation
# It's quite similar to forward mode, with the difference that we walk backwards
# over the list, and each time we see a usage of a variable yᵢ we accumulate 
# a gradient for that variable.
function derive_r(w::Wengert, x)
  ds = Dict()
  d(x) = get(ds, x, 0)
  d(x, Δ) = ds[x] = haskey(ds, x) ? addm(ds[x],Δ) : Δ
  d(lastindex(w), 1)
  for v in reverse(collect(keys(w)))
    ex = w[v]
    Δ = d(v)
    if @capture(ex, a_ + b_)
      d(a, Δ)
      d(b, Δ)
    elseif @capture(ex, a_ * b_)
      d(a, push!(w, mulm(Δ, b)))
      d(b, push!(w, mulm(Δ, a)))
    elseif @capture(ex, a_^n_Number)
      d(a, mulm(Δ, n, :($a^$(n-1))))
    elseif @capture(ex, a_ / b_)
      d(a, push!(w, mulm(Δ, b)))
      d(b, push!(w, :(-$(mulm(Δ, a))/$b^2)))
    else
      error("$ex is not differentiable")
    end
  end
  push!(w, d(x))
  return w
end

derive_r(Wengert(:(x / (1 + x^2))), :x) |> Expr
