# file for learning flux.jl
using Flux
using Plots
using LinearAlgebra

# grid for displaying functions
z = 0:0.01:2

#differentiating julia functions w.r.t a single variable
#=
f(x) = 3x^2 + 2x + 1
df(x) = gradient(f, x)[1]
d2f(x) = gradient(df, x)[1]

plot()
plot!(z, f)
plot!(z, df)
plot!(z, d2f)
=#

# if the function is multivariate, we can get the gradient w.r.t each
# variable

f(x, y) = sum((x .-y).^2)

gradient(f, [2,1], [2,0])

# when the function depends on many parameters we can use params

x = [2, 1]
y = [2, 0]

gs = gradient(params(x,y)) do
    f(x,y)
end

@show gs[x]

@show gs[y]

# Simple Models

# "synaptic weights"
W = rand(2,5)

# bias
b = rand(2)

predict(x) = W*x + b

function loss(x, y)
    ŷ = predict(x)
    sum((y .- ŷ).^2)
end

# dummy data
x, y = rand(5), rand(2)
@show loss(x, y)

# to improve the prediction we can take gradients of loss w.r.t W and b
# and perform gradient descent

gs = gradient(() -> loss(x, y), params(W,b))

# now that we have gradients we can pull them out and update W
# to train the model
Ŵ = gs[W]
W .-= 0.1 .* Ŵ

@show loss(x, y)

# more complex models than the linear regression above

W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1

W2 = rand(2,3)
b2 = rand(2)
layer2(x) = W2 * x.+ b2

model(x) = layer2(σ.(layer1(x)))

model(rand(5))

# clean this up by introducing function to return linear layers
function linear(in, out)
    W = randn(out, in)
    b = randn(out)
    x -> W*x .+ b
end

linear1 = linear(5,3)
linear2 = linear(3,2)

model(x) = linear2(σ.(linear1(x)))

# equivalently we can create a struct the explicitly represents
# the affine layer

struct Affine
    W
    b
end

Affine(in::Integer, out::Integer) =
    Affine(randn(out, in), randn(out))

# overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10,5)
a(rand(10))

# above is the Dense layer that comes with flux

# It is common to write models that look something like
#   layer1 = Dense(10, 5, σ)
#   ...
#   model(x) = layer3(layer2(layer1(x)))
#
# so for long chains it is helpful to have lists of layers

layers = [Dense(10, 5, σ), Dense(5, 2), softmax]
model(x) = foldl((x, m) -> m(x), layers, init=x)
model(rand(10))

# this is also provided by flux
model2 = Chain(
    Dense(10, 5, σ),
    Dense(5, 2),
    softmax)

model2(rand(10))

# because models are viewed as fucntions here, we can view
# this process as repeated function composition
m = Dense(5, 2) ∘ Dense(10, 5, σ)
m(rand(10))

# and Chain will work with any julia function
m = Chain(x -> x^2, x -> x+1)
m(5) # → 26




# try training sine

W = rand(2,1)

# bias
b = rand(2)

predict(x) = W*x + b

g(x) = sin(x)

function sin_loss(x)
    ŷ = model(x)
    sum((g.(x) .- ŷ).^2)
end

W1 = rand(15, 1)
b1 = rand(15)
layer1(x) = W1 * x .+ b1

W2 = rand(10,15)
b2 = rand(10)
layer2(x) = W2 * x.+ b2

W3 = rand(1,10)
b3 = rand(1)
layer3(x) = W3 * x.+ b3

model(x) = layer3(σ.(layer2(σ.(layer1(x)))))

for i = 1:100000

#    if i%10000 == 0
#        X = rand([-2*pi -pi 0 pi 2*pi])
#    else
        X = rand(0:0.001:2.5*pi, 1, 1)
#    end

    # to improve the prediction we can take gradients of loss w.r.t W and b
    # and perform gradient descent

    gs = gradient(() -> sin_loss(X), params(W1,b1, W2, b2, W3, b3))

    # now that we have gradients we can pull them out and update W
    # to train the model
    Ŵ1 = gs[W1]
    W1 .-= 0.1 .* Ŵ1

    b̂1 = gs[b1]
    b1[:] .-= 0.1 .* b̂1[:]


    Ŵ2 = gs[W2]
    W2 .-= 0.1 .* Ŵ2

    b̂2 = gs[b2]
    b2[:] .-= 0.1 .* b̂2[:]

    Ŵ3 = gs[W3]
    W3 .-= 0.1 .* Ŵ3

    b̂3 = gs[b3]
    b3[:] .-= 0.1 .* b̂3[:]
    @show sin_loss(X)
end

q = -10:0.01:10
Q = zeros(length(q))
for j = 1:length(q)
    Q[j] = model(q[j])[1]
end

plot(legend=false,size=(250,250))
plot!(q,Q)
plot!(q, g.(q[:]))
