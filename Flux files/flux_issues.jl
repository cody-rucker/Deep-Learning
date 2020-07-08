using Flux

W1 = rand(5,2)
b1 = rand(5)

W2 = rand(1, 5)
b2 = rand(1)

x = rand(2,1)

# set nerual net parameters
ps = params(W1, b1, W2, b2)

# build a simple neural net
u(x, W1, b1, W2, b2) = sum(W2*tanh.(W1*x .+ b1) .+ b2)

uₓ = (x, W1, b1, W2, b2) -> gradient(
                       x -> u(x, W1, b1, W2, b2), x)[1]

∇u = gradient(ps) do
    sum(uₓ(x, W1, b1, W2, b2))
end

uₓₓ = (x, W1, b1, W2, b2) -> gradient(
                        x -> sum(uₓ(x, W1, b1, W2, b2)), x)[1]

∇ = gradient(ps) do
    sum(uₓₓ(x, W1, b1, W2, b2))
end
