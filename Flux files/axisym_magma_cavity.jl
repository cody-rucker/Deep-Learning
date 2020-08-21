using Flux
using ProgressMeter
using ForwardDiff
using Plots
using DelimitedFiles
include("geometry.jl")
include("Adam_optimize.jl")

# solve the axisymmetric magma cavity problem in an elastic half-space

a = 2.0         # cavity radius
D = 5.0        # cavity depth
L = 5.0        # distance from remote boundary to radial center
ξ = 0.01        # resolution for randomly selected points
μ = 0.25        # μ, λ are lamé parameters
λ = 1.0

P = 1.0        # pressure along cavity wall

num_iters = 1000
batch_size = 32

r = 1.0
z = 1.0
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
"""
struct Affine
  W
  b
  f
end

Affine(in::Integer, out::Integer, f::Function=identity) = Affine(randn(out, in), randn(out), f)

(m::Affine)(x) = m.f(m.W*x .+ m.b)

mutable struct InputLayer
    Wₓ
    Wᵧ
    b
    f
end

InputLayer(in::Integer, out::Integer, f::Function=identity) = InputLayer(randn(out, 1), randn(out, 1), randn(out), f)

(m::InputLayer)(x, y) =m.f.( m.Wₓ*x .+ m.Wᵧ*y .+ m.b)

p1 = InputLayer(2, 20)
p2 = Affine(20, 1)

q1 = InputLayer(2, 20)
q2 = Affine(20, 1)
θ = Flux.params(p1.Wₓ, p1.Wᵧ, p1.b, p2.W, p2.b,
                q1.Wₓ, q1.Wᵧ, q1.b, q2.W, q2.b)
"""

Wᵣ = rand(30,1)
W𝑧 = rand(30,1)
b1 = rand(30)

W2 = rand(2,30)
b2 = rand(2)

eᵣ = [1.0; 0.0]
e𝑧 = [0.0; 1.0]

θ = Flux.params(Wᵣ, W𝑧, b1, W2, b2)

r = 1.5
z = 2.1

u_r(r, z) = sum((W2 * σ.(Wᵣ*r + W𝑧*z .+ b1) .+ b2) .* eᵣ)
u_z(r, z) = sum((W2 * σ.(Wᵣ*r + W𝑧*z .+ b1) .+ b2) .* e𝑧)
u(r, z) = (W2 * σ.(Wᵣ*r + W𝑧*z .+ b1) .+ b2)
# ∂ᵣu_r and ∂ᵣu_z
∂ᵣu(r, z) = W2 * (σ'.(Wᵣ*r + W𝑧*z .+ b1) .* Wᵣ )

@assert ∂ᵣu(r, z)[1] ≈ ForwardDiff.derivative(r->u_r(r, z), r)
@assert ∂ᵣu(r, z)[2] ≈ ForwardDiff.derivative(r->u_z(r, z), r)

#∂𝑧u_r and ∂𝑧u_z
∂𝑧u(r, z) = W2 * (σ'.(Wᵣ*r + W𝑧*z .+ b1) .* W𝑧 )

@assert ∂𝑧u(r, z)[1] ≈ ForwardDiff.derivative(z->u_r(r, z), z)
@assert ∂𝑧u(r, z)[2] ≈ ForwardDiff.derivative(z->u_z(r, z), z)

# second-order derivatives
#∂ᵣᵣu_r and ∂ᵣᵣu_z
∂ᵣᵣu(r, z) = W2 * ( σ''.(Wᵣ*r + W𝑧*z .+ b1) .* Wᵣ .* Wᵣ)

@assert ∂ᵣᵣu(r, z)[1] ≈ ForwardDiff.derivative(r-> ForwardDiff.derivative(r-> u_r(r, z), r), r)
@assert ∂ᵣᵣu(r, z)[2] ≈ ForwardDiff.derivative(r-> ForwardDiff.derivative(r-> u_z(r, z), r), r)

#∂𝑧𝑧u_r and ∂𝑧𝑧u_z
∂𝑧𝑧u(r, z) = W2 * ( σ''.(Wᵣ*r + W𝑧*z .+ b1) .* W𝑧 .* W𝑧)

@assert ∂𝑧𝑧u(r, z)[1] ≈ ForwardDiff.derivative(z-> ForwardDiff.derivative(z-> u_r(r, z), z), z)
@assert ∂𝑧𝑧u(r, z)[2] ≈ ForwardDiff.derivative(z-> ForwardDiff.derivative(z-> u_z(r, z), z), z)

∂ᵣ𝑧u(r,z) = W2 * ( σ''.(Wᵣ*r + W𝑧*z .+ b1) .* Wᵣ .* W𝑧)

@assert ∂ᵣ𝑧u(r, z)[1] ≈ ForwardDiff.derivative(r-> ∂𝑧u(r, z)[1], r)
@assert ∂ᵣ𝑧u(r, z)[2] ≈ ForwardDiff.derivative(r-> ∂𝑧u(r, z)[2], r)

# stress σ = Eε
function div_σ(r, z)
    # divergence is still probably wrong
    (∂ᵣᵣu(r, z) + ∂ᵣu(r, z)) .* [2μ + λ; μ] + ∂ᵣ𝑧u(r,z) .* (μ + λ) + ∂𝑧𝑧u(r, z) .* [μ ; 2μ + λ]
                -((2μ + λ)/r^2 * u(r,z) .* [1; 0]) + ((μ + λ)/r * ∂𝑧u(r, z) .* [1; 0] )
end

# define objective function J(r, z) accounting for boudary points
function J(mesh::Geometry)
    sum( div_σ(mesh.interior.x, mesh.interior.y).^2 +
    abs.(u(mesh.cavity.x, mesh.cavity.y) - û(mesh.cavity.x, mesh.cavity.y)).^2 +
    abs.(u(mesh.axis.x, mesh.axis.y) - û(mesh.axis.x, mesh.axis.y)).^2 +
    abs.(u(mesh.top.x, mesh.top.y) - û(mesh.top.x, mesh.top.y)).^2 +
    abs.(u(mesh.bottom.x, mesh.bottom.y) - û(mesh.bottom.x, mesh.bottom.y)).^2 +
    abs.(u(mesh.remote.x, mesh.remote.y) - û(mesh.remote.x, mesh.remote.y)).^2)
end

# initialize Adam objects to store optimization parameters
Wr = Adam(Wᵣ, Wᵣ)
Wz = Adam(W𝑧, W𝑧)
b₁ = Adam(b1, b1)
W₂ = Adam(W2, W2)
b₂ = Adam(b2, b2)


#@showprogress "Training..."
for i = 1:num_iters


    for j = 1:batch_size
          # generate random points on domain boundary and interior
          mesh = Geometry(a, D, L, ξ)

          ∇u = gradient(θ) do
              J(mesh)
          end

          # Adam optimisation
          Adam_update!( Wᵣ, Wr, ∇u[Wᵣ], i)
          Adam_update!( W𝑧, Wz, ∇u[W𝑧], i)
          Adam_update!( b1, b₁, ∇u[b1], i)
          Adam_update!( W2, W₂, ∇u[W2], i)
          Adam_update!( b2, b₂, ∇u[b2], i)

         @show J(mesh)
    end

end


#=
Wr_r = rand(30, 1)
Wr_z = rand(30, 1)
br1 = rand(30)

Wr2 = rand(1, 30)
br2 = rand(1)

Wz_r = rand(30, 1)
Wz_z = rand(30, 1)
bz1 = rand(30)

Wz2 = rand(1, 30)
bz2 = rand(1)
θ = Flux.params(Wr_r, Wr_z, br1, Wr2, br2,
                Wz_r, Wz_z, bz1, Wz2, bz2)

Σr(r, z) = Wr_r * r + Wr_z * z .+ br1
Σz(r, z) = Wz_r * r + Wz_z * z .+ bz1
ur(r, z) = sum(Wr2 * σ.(Σr(r, z)) .+ br2)
uz(r, z) = sum(Wz2 * σ.(Σz(r, z)) .+ bz2)

# first partial derivatives
∂ᵣur(r, z) = sum(Wr2 * (σ'.(Σr(r, z)) .* Wr_r))
∂𝑧ur(r, z) = sum(Wr2 * (σ'.(Σr(r, z)) .* Wr_z))
∂ᵣuz(r, z) = sum(Wz2 * (σ'.(Σz(r, z)) .* Wz_r))
∂𝑧uz(r, z) = sum(Wz2 * (σ'.(Σz(r, z)) .* Wz_z))

# check first derivative calculations with ForwardDiff
#=
@assert ∂ᵣur(r, z) ≈ ForwardDiff.derivative(r-> ur(r, z), r)
@assert ∂𝑧ur(r, z) ≈ ForwardDiff.derivative(z-> ur(r, z), z)
@assert ∂ᵣuz(r, z) ≈ ForwardDiff.derivative(r-> uz(r, z), r)
@assert ∂𝑧uz(r, z) ≈ ForwardDiff.derivative(z-> uz(r, z), z)
=#

# second-order derivatives
∂ᵣᵣur(r, z) = sum(Wr2 * (σ''.(Σr(r, z)) .* Wr_r .* Wr_r))
∂𝑧𝑧ur(r, z) = sum(Wr2 * (σ''.(Σr(r, z)) .* Wr_z .* Wr_z))
∂ᵣᵣuz(r, z) = sum(Wz2 * (σ''.(Σz(r, z)) .* Wz_r .* Wz_r))
∂𝑧𝑧uz(r, z) = sum(Wz2 * (σ''.(Σz(r, z)) .* Wz_z .* Wz_z))

# note that ∂𝑧ᵣur = ∂ᵣ𝑧ur and ∂𝑧ᵣuz = ∂ᵣ𝑧uz
∂𝑧ᵣur(r, z) = sum(Wr2 * (σ''.(Σr(r, z)) .* Wr_z .* Wr_r))
∂𝑧ᵣuz(r, z) = sum(Wz2 * (σ''.(Σz(r, z)) .* Wz_z .* Wz_r))

# check second derivative calculations with ForwardDiff
#=
@assert ∂ᵣᵣur(r, z) ≈ ForwardDiff.derivative(r-> ForwardDiff.derivative(r-> ur(r, z), r), r)
@assert ∂𝑧𝑧ur(r, z) ≈ ForwardDiff.derivative(z-> ForwardDiff.derivative(z-> ur(r, z), z), z)
@assert ∂ᵣᵣuz(r, z) ≈ ForwardDiff.derivative(r-> ForwardDiff.derivative(r-> uz(r, z), r), r)
@assert ∂𝑧𝑧uz(r, z) ≈ ForwardDiff.derivative(z-> ForwardDiff.derivative(z-> uz(r, z), z), z)
@assert ∂𝑧ᵣur(r, z) ≈ ForwardDiff.derivative(z-> ForwardDiff.derivative(r-> ur(r, z), r), z)
@assert ∂𝑧ᵣuz(r, z) ≈ ForwardDiff.derivative(z-> ForwardDiff.derivative(r-> uz(r, z), r), z)
=#
# ∇⋅σ, divergence of stress written in terms of displacements

function div_σ(r, z)
    a = (2μ + λ) * (∂ᵣᵣur(r, z) + ∂𝑧ᵣuz(r, z) + (1/r) * (∂ᵣur(r, z) -
                                                ur(r, z)/r)) - μ * ∂𝑧ᵣuz(r, z)
    b = 0#(2μ + λ) * ∂𝑧𝑧uz(r, z) + (λ + μ) * (∂𝑧ᵣur(r, z) + (1/r)*∂𝑧ur(r, z)) +
                                            #    (μ/r) * ∂ᵣuz(r, z) + μ*∂ᵣᵣuz(r, z)
    return sqrt(a^2 + b^2)
end

# define objective function J(r, z) accounting for boudary points
function J(mesh::Geometry)
    sum( div_σ(mesh.interior.x, mesh.interior.y)^2 +
    abs.(ur(mesh.cavity.x, mesh.cavity.y) - ûr(mesh.cavity.x, mesh.cavity.y)).^2 +
    abs.(uz(mesh.cavity.x, mesh.cavity.y) - ûz(mesh.cavity.x, mesh.cavity.y)).^2 +
    abs.(ur(mesh.axis.x, mesh.axis.y) - ûr(mesh.axis.x, mesh.axis.y)).^2 +
    abs.(uz(mesh.axis.x, mesh.axis.y) - ûz(mesh.axis.x, mesh.axis.y)).^2 +
    abs.(ur(mesh.top.x, mesh.top.y) - ûr(mesh.top.x, mesh.top.y)).^2 +
    abs.(uz(mesh.top.x, mesh.top.y) - ûz(mesh.top.x, mesh.top.y)).^2 +
    abs.(ur(mesh.bottom.x, mesh.bottom.y) - ûr(mesh.bottom.x, mesh.bottom.y)).^2 +
    abs.(uz(mesh.bottom.x, mesh.bottom.y) - ûz(mesh.bottom.x, mesh.bottom.y)).^2 +
    abs.(ur(mesh.remote.x, mesh.remote.y) - ûr(mesh.remote.x, mesh.remote.y)).^2 +
    abs.(uz(mesh.remote.x, mesh.remote.y) - ûz(mesh.remote.x, mesh.remote.y)).^2 )
end

# initialize Adam objects to store optimization parameters
Wrᵣ = Adam(Wr_r, Wr_r)
Wr𝑧 = Adam(Wr_z, Wr_z)
br₁ = Adam(br1, br1)
Wr₂ = Adam(Wr2, Wr2)
br₂ = Adam(br2, br2)

Wzᵣ = Adam(Wz_r, Wz_r)
Wz𝑧 = Adam(Wz_z, Wz_z)
bz₁ = Adam(bz1, bz1)
Wz₂ = Adam(Wz2, Wz2)
bz₂ = Adam(bz2, bz2)

#@showprogress "Training..."
for i = 1:num_iters
    for j = 1:batch_size
          # generate random points on domain boundary and interior
          mesh = Geometry(a, D, L, ξ)

          ∇u = gradient(θ) do
              J(mesh)
          end

          # Adam optimisation
          Adam_update!(Wr_r, Wrᵣ, ∇u[Wr_r], i)
          Adam_update!(Wr_z, Wr𝑧, ∇u[Wr_z], i)
          Adam_update!(br1, br₁, ∇u[br1], i)
          Adam_update!(Wr2, Wr₂, ∇u[Wr2], i)
          Adam_update!(br2, br₂, ∇u[br2], i)

          Adam_update!(Wz_r, Wzᵣ, ∇u[Wz_r], i)
          Adam_update!(Wz_z, Wz𝑧, ∇u[Wz_z], i)
          Adam_update!(bz1, bz₁, ∇u[bz1], i)
          Adam_update!(Wz2, Wz₂, ∇u[Wz2], i)
          Adam_update!(bz2, bz₂, ∇u[bz2], i)

         @show J(mesh)
    end

end


#=
writedlm("NN_weights/Wx.csv", Wₓ, ',')
writedlm("NN_weights/Wy.csv", Wᵧ, ',')
writedlm("NN_weights/.csv", A, ',')
=#
=#
