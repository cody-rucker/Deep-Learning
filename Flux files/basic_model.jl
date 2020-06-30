using Flux
using Plots
using LinearAlgebra

#=
 establish model using Flux built in Dense layer and Chain function
 which composes layers:
               (layer2 ∘ layer1)(x)
 this also has the added benefit that the output type is simply
 an array, not an array of arrays ... of arrays.
 For now, take input and output to each be 1×1 arrays
=#
let
ins = 1
outs = 1

# set some number that dictacts coarseness of random point generation
Δξ = 0.001

# learning rate
α = 0.07

# function to be learned by the machine
f(x) = sin(x)

function loss(x)
        ŷ = model(x)
        sum((f.(x) .- ŷ).^2)
end

# feed-forward network
model = Chain(
        Dense(ins, 20, tanh),
        Dense(20, 15, tanh),
        Dense(15, outs))

z = -3π:0.01:3π
z1 = [0.0]
N = 500
Z1 = zeros(length(z), N +1)
k = 1
M = 90000
for i = 1:M
        ξ = rand(-2π:Δξ:2π, 1, 1)

        # find gradient of loss w.r.t weights and biases
        #=gs = gradient(() -> loss(ξ), params(ξ, model[1].W, model[1].b,
                                            model[2].W, model[2].b,
                                            model[3].W, model[3].b))=#
        #= gs = gradient(params( model[1].b,
                                             model[2].W, model[2].b,
                                             model[3].W, model[3].b)) do
                                                     loss(ξ)
                                             end=#
         gs = gradient(() -> loss(ξ), params(model))

        for i = 1:length(model)-1
                model[i].W .-= α .* gs[model[i].W]
                model[i].b .-= α .* gs[model[i].b]
        end
        #@show loss(ξ)

        if i%(M/N) == 0
                for j = 1:length(z)
                        z1 .= z[j]
                        Z1[j, k] = model(z1)[1]
                end
                k += 1
        end
end

for n = 1:N
        p = plot(z, Z1[:,n], size=(1000,750), lw=3, legend=:bottomright, label="network")
            plot!(z, f.(z), lw=3, label="exact")
            plot!([0, π, 2π, -π, -2π], seriestype="vline", label="", color="green4", lw=2)
            #plot!([π], seriestype="vline", label="", color="green4")
        ylims!((-1,1))
        display(p)
end
end

#Z = zeros(length(z))

#for i = 1:length(z)
#        z1 .= z[i]
#        Z[i] = model(z1)[1]
#end

#plot(size=(250,250), legend=false)
#plot!(z, f.(z))
#plot!(z, Z)
