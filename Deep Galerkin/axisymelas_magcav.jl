using Flux
using ProgressMeter
using ForwardDiff
using Plots
using DelimitedFiles
include("geometry.jl")
include("NNets.jl")

# solve the axisymmetric magma cavity problem in an elastic half-space

a = 2.0         # cavity radius
D = 5.0        # cavity depth
L = 5.0        # distance from remote boundary to radial center
ξ = 0.01        # resolution for randomly selected points
μ = 0.25        # μ, λ are lamé parameters
λ = 1.0

P = 1.0        # pressure along cavity wall

num_iters = 100 # number of iterations through training algorithm
N = 10          # hidden dimension
#=
 use the known solution to the magmacavity problem in an elastic full-
 space as a manufactured solution
 uᵣ = P*a³r / 4μ(r²+z²)^(3/2)
 u𝑧 = P*a³z / 4μ(r²+z²)^(3/2)
 r = ρcosϕ
 z = ρsinϕ
=#
#r(ϕ, ρ) = ρ * cos(ϕ)
#z(ϕ, ρ) = ρ * sin(ϕ)

ûr(r, z) = P * a * r / 4 * μ * ( r^2 + z^2)^(3/2)
ûz(r, z) = P * a * z / 4 * μ * ( r^2 + z^2)^(3/2)

û(r, z) = (P * a)  / (4 * μ * ( r^2 + z^2)^(3/2)) .* [r; z]

u = NeuralNet(N, 2, 2)

uₓ = FirstNetDerivative(u, "x₁")
uᵧ = FirstNetDerivative(u, "x₂")


uₓₓ = SecondNetDerivative(u, "x₁", "x₁")
uᵧᵧ = SecondNetDerivative(u, "x₂", "x₂")
uₓᵧ = SecondNetDerivative(u, "x₁", "x₂")

function div_σ()
    threshold = 1e-15
#=
a = 1.5
L = 5.0
D = 5.0
w = true
x_grid = 0:0.05:L
y_grid = -D:0.1:D

Z = zeros(length(x_grid), length(y_grid))

for i = 1:length(x_grid)
    for j = 1:length(y_grid)
        if (sqrt(x_grid[j]^2 + y_grid[i]^2) < a)
            Z[i, j] = 0
        else
            Z[i, j] = û(x_grid[j], y_grid[i])[2]
        end
    end
end

pyplot()
plot(x_grid, y_grid, Z[:,:], st=:contour, xlabel="x", ylabel="y")
=#
