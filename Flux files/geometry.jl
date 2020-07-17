using Distributions
using Plots

mutable struct bdry
    θ::Float64
    ρ::Float64
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


# visualize the scatter mesh
#=
p = scatter(proj=:polar, legend= :false, showaxis= :true,
            grid=:true, markercolor=:red4, markersize=0.5,
            alpha=1 , markershape = :circle, size=(1000,750))
for i = 1:10000
    v = Geometry(2.0, 10.0, 10.0)

    x = [ v.cavity.θ,
          v.axis.θ,
          v.top.θ,
          v.bottom.θ,
          v.remote.θ,
          v.interior.θ ]

    y = [ v.cavity.ρ,
          v.axis.ρ,
          v.top.ρ,
          v.bottom.ρ,
          v.remote.ρ,
          v.interior.ρ ]

    scatter!(x, y, proj=:polar, legend= :false, showaxis= :true,
                   grid=:true, markercolor=:red4, markersize=0.5,
                   alpha=1 , markershape = :circle, size=(1000,750))

end

#display(p)
=#
