using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Plots
using Test
include("NeuralNetDiffEq.jl/src/NeuralNetDiffEq.jl")

# 2D Poisson problem

@parameters x y θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq = Dxx(u(x, y, θ)) + Dyy(u(x, y, θ)) ~ -sin(pi*x) * sin(pi*y)

# Boundary conditions
bcs = [u(0,y) ~ 0.f0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.f0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
discretization = PhysicsInformedNN(dx)

# Neural network and optimizer
opt = Flux.ADAM(0.02)
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)
alg = NNDE(chain,opt,autodiff=false)
phi,res  = solve(prob,alg,verbose=true, maxiters=5000)

xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)
u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))

@test u_predict ≈ u_real atol = 0.3

p1 =plot(xs, ys, u_predict, st=:surface);
p2 = plot(xs, ys, u_real, st=:surface);
plot(p1,p2)
