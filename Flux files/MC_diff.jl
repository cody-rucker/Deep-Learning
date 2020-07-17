using Distributions
using Flux
using Plots

function MC_diff(f::Function, y::Float64; Δ::Float64 = 0.0001,
                    num_iters::Int64 = 100000)
    # num_itersonte Carlo methods for approximating derivatives
    μ = 0           # mean of the truncated normal dist.
    σ = 1           # standard deviation of the truncanted normal
    a = 0.0         # truncated lower bound
    b = Δ           # truncated upper bound

    # construct the distribution type
    d = Truncated(Normal(μ, σ), a, b)
    #=
    using StatsPlots
    plot(d) # verify normal distribution with mean 0
    =#

    # simple function for testing
    #f(x) = sin(π*x) + cos(π*x)

    # "diffusion"?
    ς(x) = 1.0

    df = 0.0    # initialize derivative estimate

    for i = 1:num_iters
        ε = rand(d, 1)[1]
        df += (f(y + ε) - ς(y)*f(y)) / Δ
        #df += ((f(y + ς(y)*ε/2) - f(y - ς(y) *ε/2)) / Δ)
    end
    return (2/num_iters) * df
end

function test_MC_diff(f, Δ, N)
    #Δ = 0.01
    #N = 1000

    df(x) = MC_diff(f, x)#, Δ = Δ, num_iters = N)
    δf(x) = gradient(f, x)[1]

    x = 0:0.001:1
    dz = zeros(length(x))
    δz = zeros(length(x))
    err = zeros(length(x))

    for i = 1:length(x)
        dz[i] = df(x[i])
        δz[i] = δf(x[i])
        err[i] = dz[i] - δz[i]
    end

    p1 = plot(x, dz[:], size=(1000, 750), lw=1.5,
                            legend=:topright, label = "MC")
         plot!(x, δz[:], label= "AD")
    p2 = plot(x, err[:])
    p = plot(p1, p2, layout = (2,1),  size=(1000, 750))
#    return dz, δz, err, x
end
