using Flux
using ForwardDiff
using ProgressMeter
using Plots
using IJulia
include("Adam_optimise.jl")


M = 1500
batch_size = 32

uexact(x, y, t) = sin(2*π*(x - t))
# initial condition with exact solution u(x) = sin(2π(x-t))
η(x, y) = sin(2*π*x)
γ(x, y) = -2*π * cos(2*π*x)

# boundary condition at y = 0 and y = 1
yb₀(x, t) = sin(2*π*(x - t))
yb₁(x, t) = sin(2*π*(x - t))

# boundary conditions at x = 0 and x = 1
xb₀(y, t) =  - sin(2*π*t)
xb₁(y, t) = sin(2*π*(1-t))
######################################################

gᵪ(x, y, t) = 2*π^2 * (sin(π*(x-t)) + sin(π*(y-t)))

Wₓ = rand(20, 1)
Wₜ = rand(20, 1)
Wᵧ = rand(20, 1)
b1 = rand(20)

W2 = rand(25, 20)
b2 = rand(25)

W3 = rand(1, 25)
b3 = rand(1)
θ = Flux.params(Wₓ, Wᵧ, Wₜ, b1, W2, b2, W3, b3)

u(x, y,  t) = sum(W3 * σ.(
              W2 * σ.(
              (Wₓ*x + Wᵧ*y .+ Wₜ*t) .+ b1) .+ b2) .+ b3)

# sigmoid derivative
dσ(x) =  σ(x) * (1 - σ(x))
d²σ(x) = dσ(x) - 2* dσ(x)*σ(x)

# first-order derivatives
uₓ(x, y, t) = sum( W3 * ( dσ.( W2 * σ.(Wₓ*x .+ Wᵧ.+ Wₜ*t .+ b1) .+ b2) .*
                       (W2 * (dσ.(Wₓ*x .+ Wᵧ.+ Wₜ*t .+ b1) .* Wₓ)) ))

uₜ(x, y, t) = sum( W3 * ( dσ.( W2 * σ.(Wₓ*x .+ Wᵧ.+ Wₜ*t .+ b1) .+ b2) .*
                      (W2 * (dσ.(Wₓ*x .+ Wᵧ.+ Wₜ*t .+ b1) .* Wₜ)) ))

uᵧ(x, y, t) = sum( W3 * ( dσ.( W2 * σ.(Wₓ*x .+ Wᵧ.+ Wₜ*t .+ b1) .+ b2) .*
                      (W2 * (dσ.(Wₓ*x .+ Wᵧ.+ Wₜ*t .+ b1) .* Wᵧ)) ))


# verify accuracy of derivatives with forward AD
x = 1.0
y = 1.0
t = 2.0
@assert uₓ(x, y, t) ≈ ForwardDiff.derivative(x -> u(x, y,t), x)
@assert uᵧ(x, y, t) ≈ ForwardDiff.derivative(y -> u(x, y,t), y)
@assert uₜ(x, y, t) ≈ ForwardDiff.derivative(t -> u(x, y,t), t)
#uₓ(x, y, t) = sum(W2 * (dσ.(Wₓ*x .+ Wₜ*t .+ b1) .* Wₓ))
#uₜ(x, y, t) = sum(W2 * (dσ.(Wₓ*x .+ Wₜ*t .+ b1) .* Wₜ))

# second-order derivatives ( these ones arent as accurate check work later)
function uₓₓ(x, y,  t)
    Σ = Wₓ*x .+ Wᵧ*y .+ Wₜ*t .+ b1

    a = d²σ.(W2 * σ.(Σ) .+ b2) .* (W2 * (dσ.(Σ) .* Wₓ)) .* (W2 * (dσ.(Σ) .* Wₓ))

    b = dσ.( W2 * σ.(Σ) .+ b2) .* (W2 * (d²σ.(Σ) .* Wₓ .* Wₓ) )

    return sum(W3 * (a .+ b))
end

function uₜₜ(x, y, t)
    Σ = Wₓ*x .+ Wᵧ*y .+ Wₜ*t .+ b1

    a = d²σ.(W2 * σ.(Σ) .+ b2) .* (W2 * (dσ.(Σ) .* Wₜ)) .* (W2 * (dσ.(Σ) .* Wₜ))

    b = dσ.( W2 * σ.(Σ) .+ b2) .* (W2 * (d²σ.(Σ) .* Wₜ .* Wₜ) )

    return sum(W3 * (a .+ b))
end

function uᵧᵧ(x, y, t)
    Σ = Wₓ*x .+ Wᵧ*y .+ Wₜ*t .+ b1

    a = d²σ.(W2 * σ.(Σ) .+ b2) .* (W2 * (dσ.(Σ) .* Wᵧ)) .* (W2 * (dσ.(Σ) .* Wᵧ))

    b = dσ.( W2 * σ.(Σ) .+ b2) .* (W2 * (d²σ.(Σ) .* Wᵧ .* Wᵧ) )

    return sum(W3 * (a .+ b))
end

#=
function uᵧᵧ(x, y, t)
    Σ = Wₓ*x .+ Wᵧ*y .+ Wₜ*t .+ b1

    a = W3 *( d²σ.( W2 * σ.(Σ)) .*
            (W2 * (dσ.(Σ) .* Wᵧ)).^2)

    b = W3 * (dσ.(W2 * σ.(Σ) .+ b2) .*
              (W2 * (d²σ.(Σ) .* Wᵧ.^2)))
    return sum(a + b)
end
=#
#uₓₓ(x, y, t) = sum(W2 * (d²σ.(Wₓ*x .+ Wₜ*t .+ b1) .* Wₓ .* Wₓ))
#uₜₜ(x, y, t) = sum(W2 * (d²σ.(Wₓ*x .+ Wₜ*t .+ b1) .* Wₜ .* Wₜ)) #slightly disagrees with autograd...

#=
Define a physics informed neural net
            f := uₜₜ + N[u]
and proceed by approximating u(t,x) with a deep neural network
=#
function f(x, y, t)
    return uₜₜ(x, y, t)- uₓₓ(x, y, t) - uᵧᵧ(x, y, t)
end

# cost function to enforce PDE conditions
function cost(x, y, t,
              x̂, ŷ, t̂,
              ẋ, ẏ, ṫ,
              x₀, y₀, t₀,
              x₁, y₁, t₁,
              x̄₀, ȳ₀, t̄₀,
              x̄₁, ȳ₁, t̄₁)
    sum(abs.(f(x, y, t)).^2 +                     # enforce structure of the PDE
        abs.(u(x̂, ŷ, t̂) .- η.(x̂, ŷ)).^2 +            # initial displacement
        abs.(uₜ(ẋ, ẏ, ṫ) .- γ.(ẋ, ẏ)).^2 +            # initial velocity
        abs.(u(x₀, y₀, t₀) .- xb₀.(y₀,t₀)).^2 +         # b.c at x=0
        abs.(u(x₁, y₁, t₁) .- xb₁.(y₁,t₁)).^2 +        # b.c at x=1
        abs.(u(x̄₀, ȳ₀, t̄₀) .- yb₀.(x̄₀,t̄₀)).^2 +         # b.c at y=0
        abs.(u(x̄₁, ȳ₁, t̄₁) .- yb₁.(x̄₁,t̄₁)).^2   )      # b.c at y= 1
end

# initialize Adam objects to store optimization parameters
Wx = Adam(Wₓ, Wₓ)
Wt = Adam(Wₜ, Wₜ)
Wy = Adam(Wᵧ, Wᵧ)
b₁ = Adam(b1, b1)

W₂ = Adam(W2, W2)
b₂ = Adam(b2, b2)

W₃ = Adam(W3, W3)
b₃ = Adam(b3, b3)

# training loop: Adam optimisation
@showprogress "Training..." for n = 1:M
    sleep(0.1)
    for i = 1:batch_size

        x = rand(0:0.001:1, 1, 1)[1]          # random x∈   Ω×[0,T]
        y = rand(0:0.001:1, 1, 1)[1]
        t = rand(0:0.001:1, 1, 1)[1]

        x̂ = rand(0:0.001:1, 1, 1)[1]          # random x∈   Ω×{0}
        ŷ = rand(0:0.001:1, 1, 1)[1]
        t̂ = 0.0

        ẋ = rand(0:0.001:1, 1, 1)[1]          # random x∈   Ω×{0}
        ẏ = rand(0:0.001:1, 1, 1)[1]
        ṫ = 0.0

        x₀ =0.0                              # random x∈ {0}×[0,T]
        y₀ = rand(0:0.001:1.0, 1, 1)[1]
        t₀ = rand(0:0.001:1.0, 1, 1)[1]

        x₁ = 1.0                             # random x∈ {1}×[0,T]
        y₁ = rand(0:0.001:1.0, 1, 1)[1]
        t₁ = rand(0:0.001:1.0, 1, 1)[1]

        x̄₀ = rand(0:0.001:1.0, 1, 1)[1]
        ȳ₀ = 0.0
        t̄₀ = rand(0:0.001:1.0, 1, 1)[1]

        x̄₁ = rand(0:0.001:1.0, 1, 1)[1]
        ȳ₁ = 1.0
        t̄₁ = rand(0:0.001:1.0, 1, 1)[1]

        ∇u = gradient(θ) do
            cost(x, y, t, x̂, ŷ, t̂,ẋ, ẏ, ṫ,
                 x₀, y₀, t₀, x₁, y₁, t₁,
                 x̄₀, ȳ₀, t̄₀, x̄₁, ȳ₁, t̄₁)
        end

        # Adam optimisation
        Adam_update!(Wₓ, Wx, ∇u[Wₓ], i)
        Adam_update!(Wₜ, Wt, ∇u[Wₜ], i)
        Adam_update!(Wᵧ, Wy, ∇u[Wᵧ], i)
        Adam_update!(b1, b₁, ∇u[b1], i)
        Adam_update!(W2, W₂, ∇u[W2], i)
        Adam_update!(b2, b₂, ∇u[b2], i)
        Adam_update!(W3, W₃, ∇u[W3], i)
        Adam_update!(b3, b₃, ∇u[b3], i)

    # Stochastic gradient descent (SGD)
    #    @inbounds for j = 1:length(θ)
    #        θ[j] .-= α .* ∇u[θ[j]]
    #    end
    #    @show cost(x, t,  x̂, t̂, ẋ, ṫ, x₀, t₀, x₁, t₁)
    end
end


r = 0:0.01:1
s = 0:0.01:1
τ = 0:0.01:1


Z = zeros(length(r), length(s), length(τ))
V = zeros(length(r), length(s), length(τ))
E = zeros(length(r), length(s), length(τ))
for i = 1:length(τ)
    for j = 1:length(s)
        for k = 1:length(r)
            Z[k, j, i] = u(r[k], s[j], τ[i])
            V[k, j, i] = uexact(r[k], s[j], τ[i])
            E[k, j, i] = V[k, j, i] - Z[k, j, i]
        end
    end
end
#=
plt = plot3d(st=:surface, xlim=(-0,1), ylim=(0,1), zlim=(-1,1),
                title = "wave", marker = 2,
                xlabel="x", ylabel="y", zlabel="z",
                size=(1920, 1080),
                label="")
anim = @animate
for i = 1:length(τ)

        p = plot3d(r, s, Z[:,:,i]), st=:surface, size=(1920, 1080), show=true)# xlim=(-0,1), ylim=(0,1), zlim=(-1,1),
                        #title = "wave", marker = 2,
                        #xlabel="x", ylabel="y", zlabel="z",
                        #size=(1920, 1080),
                        #label="")
        display(p)

    end
    =#

pyplot()
#plot(legend=true, size=(1920, 1080)
#anim = @animate
#for i = 1:length(τ)
        #plot(x,exact(x[:],t[i]))
        #plot(x[:], [U[:,i], exact(x[:], t[i])])
        #p1 = plot(r, s,Z[:,:,i], st=:surface, title="exact")
        #p2 = plot(x, [U1[:,i],U2[:,i]], label=labels, shape=markershapes)
        #p = plot(p1,p2, layout=(1,2))
#        display(pp)
#    end
#    gif(anim, "anim_fps30.gif", fps = 30)

#pyplot()

#=
anim = @animate for i = 1:length(τ)
    p1 = plot(r, s, Z[:, :, i], st=:surface, title="network")
    p2 = plot(r, s, V[:, :, i], st=:surface, title="exact")
    p = plot(p1,p2, layout=(1,2), size=(1000, 750))
    #display(p)
    sleep(0.1)
end
gif(anim, "anim_fps30.gif", fps = 30)
=#
    #=
    p1 = surface(r, s, uexact.(r[:], s[:], τ[1]), title="exact")
    p2 = surface(r, s, u.(r[:], s[:], τ[1]), title="approximation")
    plot(p1,p2, layout=(1,2))
    =#
#=
v = 0:0.001:1
xfine = 0:0.001:1
Z = zeros(length(xfine), length(v))

uexact(x, y, t) = sin(2*π*(x-t))
err(x, y, t) = uexact.(x, y, t) - u(x, y, t)# θ)

@inbounds for i = 1:length(v)
    for j = 1:length(xfine)
        Z[j,i] = u(xfine[j], v[i])
    end
end

@inbounds for i = 1:2:length(v)
    p1 = plot(xfine, u.(xfine[:], v[i]), size=(1000, 750), ylims=(-1.2, 1.2), lw=1.5,
                            legend=:topright, label = "network")

    plot!(xfine, uexact.(xfine[:], v[i]), label="exact")

    p2 = plot(xfine, err.(xfine[:], v[i]), ylims = (-1.2, 1.2), label="error",
                            color=:red)

    p = plot(p1, p2, layout = (2,1), size=(1000, 750))

    display(p)
end
=#
