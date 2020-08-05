using Distributions
using Plots

#=
mutable struct bdry
    θ::Float64
    ρ::Float64
end
=#

mutable struct bdry
    x::Float64
    y::Float64
end


mutable struct Geometry
    r::Float64      # cavity radius
    D::Float64      # cavity depth
    L::Float64      # domain length
    ξ::Float64      # domain resolution
    ϕ::Float64      # angle associated with domain corner
    R::Float64      # max possible distance from cavity center
    cavity::bdry
    axis::bdry
    top::bdry
    bottom::bdry
    remote::bdry
    interior::bdry

end

function Geometry(r::Float64, D::Float64, L::Float64, ξ::Float64 = 0.001)
    ϕ = atan(D/L)
    R = sqrt( L^2 + D^2)

    # random point along cavity wall boundary
    ρ, θ = (r, rand(π/2 : -ξ : -π/2))
    cavity = bdry(ρ*cos(θ), ρ*sin(θ))

    # random point along axis boundary
    ρ, θ = (rand(r: ξ : D), rand((π/2, -π/2)))
    axis = bdry(ρ*cos(θ), ρ*sin(θ))

    # randome point along boundary at Earth's surface
    top = bdry(rand(0 : ξ : L), D )

    # random point along boundary at depth
    bottom = bdry(rand(0 : ξ : L), -D)

    # random point along remote boundary
    remote = bdry(L, rand(-D : ξ : D ))

    # random interior point
    ρ, θ = (0.0, rand((π/2 : -ξ : -π/2)))
    if -ϕ <= θ <= ϕ
        μ = r
        σ = 4.0
        a = r
        b = L / cos(θ)
        ρ = rand( Truncated(Normal(μ, σ), a, b) )
    else
        μ = r
        σ = 4.0
        a = r
        b = D / sin(abs(θ))
        ρ = rand( Truncated(Normal(μ, σ), a, b) )
    end
    interior = bdry(ρ*cos(θ), ρ*sin(θ))

    Geometry(r, D, L, ξ, ϕ, R, cavity, axis, top, bottom, remote, interior)
end

#= cylindrical "grid" generation

function Geometry(r::Float64, D::Float64, L::Float64, ξ::Float64 = 0.001)
    ϕ = atan(D/L)
    R = sqrt( L^2 + D^2)

    cavity = bdry(rand(π/2 : -ξ : -π/2), r)

    axis = bdry(rand((π/2, -π/2)), rand(r: ξ : D))

    top = bdry(rand(π/2 : -ξ : ϕ), 0.0 )
    top.ρ = D / sin(top.θ)

    bottom = bdry(rand(-π/2 : ξ : -ϕ), 0.0)
    bottom.ρ = -D / sin(bottom.θ)

    remote = bdry(rand(-ϕ : ξ : ϕ), 0.0)
    remote.ρ = L / cos(remote.θ)

    interior = bdry(rand(π/2 : -ξ : -π/2), 0.0)

    if -ϕ <= interior.θ <= ϕ
        μ = r
        σ = 4.0
        a = r
        b = L / cos(interior.θ)
        interior.ρ = rand( Truncated(Normal(μ, σ), a, b) )
    else
        μ = r
        σ = 4.0
        a = r
        b = D / sin(abs(interior.θ))
        interior.ρ = rand( Truncated(Normal(μ, σ), a, b) )
    end

    Geometry(r, D, L, ξ, ϕ, R, cavity, axis, top, bottom, remote, interior)
end
=#

# visualize the scatter mesh
#=
p = scatter(legend= :false, showaxis= :true,
            grid=:true, markercolor=:red4, markersize=0.5,
            alpha=1 , markershape = :circle, size=(1000,750))
for i = 1:1000
    v = Geometry(2.0, 10.0, 5.0)

    x = [ v.cavity.x,
          v.axis.x,
          v.top.x,
          v.bottom.x,
          v.remote.x,
          v.interior.x ]

    y = [ v.cavity.y,
          v.axis.y,
          v.top.y,
          v.bottom.y,
          v.remote.y,
          v.interior.y ]

    scatter!(x, y, legend= :false, showaxis= :true, xlims=(0, 10),
                   grid=:true, markercolor=:red4, markersize=0.5,
                   alpha=1 , markershape = :circle, size=(750,750))

end

#display(p)
=#
