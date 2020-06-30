using Plots
using Random

r = 1
D = 5
L = 5
ϕ = atan(D/L)

Δξ = 0.025
Δη = Δξ

ξ = 0:Δξ: 1
η = 0:Δη: 1

n = length(ξ)
m = length(η)

R = sqrt(D^2 + L^2)

χ = 1000     # number of random points to grab

#pyplot()
#scatter()
#scatter( proj=:polar, legend= :false, showaxis= :true, m=2, grid=:true, markercolor=:black, markersize=0.1,alpha=.5 , markershape = :hexagon)
#scatter!(size=(700,700))

# cavity points
y = rand(π/2:-0.01:-π/2, χ, 1)
x = r*ones(length(y))

# z-axis points
ŷ = rand((π/2, -π/2), χ, 1)
x̂ = rand(r:0.01:D, length(y), 1)

# boundary at Earth's surface (z = D)
yˢ = rand((π/2:-0.01:π/4), χ, 1)
xˢ = zeros(length(yˢ))

for i = 1:length(yˢ)
    xˢ[i] = rand(r:0.01:(D/sin(yˢ[i])), 1, 1)[1]
    xˢ[i] = R + (R-r)* atan((xˢ[i] - R)) / atan(R)
    #xˢ[i] = D/sin(yˢ[i])
end

# boundary at depth z = -D
yᴰ = rand((-π/2:0.01:-π/4), χ, 1)
xᴰ = zeros(length(yᴰ))

for i = 1:length(yᴰ)
    #xᴰ[i] = rand(r:0.01:(-D/sin(yᴰ[i])), 1, 1)[1]
    xᴰ[i] = -D/sin(yᴰ[i])
end

# remote boundary at r = L
yᴿ = rand( π/4 :-0.01: -π/4, χ, 1)
xᴿ = zeros(length(yᴰ))

for i = 1:length(yᴰ)
    #xᴿ[i] = rand(r:0.01:(L/cos(yᴿ[i])), 1, 1)[1]
    xᴿ[i] = L/cos(yᴿ[i])
end

#scatter!(y, x, proj=:polar)
scatter(y, x, proj=:polar, legend= :false, showaxis= :true, grid=:true, markercolor=:red4, markersize=0.1,alpha=.7 , markershape = :circle)
scatter!(ŷ, x̂, proj=:polar, markercolor= :blue4, markersize=0.1,alpha=.7 , markershape = :circle)
scatter!(yˢ, xˢ, proj=:polar, legend= :false, showaxis= :true, grid=:true, markercolor=:green4, markersize=0.1,alpha=.7 , markershape = :circle)
scatter!(yᴰ, xᴰ, proj=:polar, legend= :false, showaxis= :true, grid=:true, markercolor=:purple2, markersize=0.1,alpha=0.7 , markershape = :circle)
scatter!(yᴿ, xᴿ, proj=:polar, legend= :false, showaxis= :true, grid=:true, markercolor=:orange2, markersize=0.1,alpha=1 , markershape = :circle)
