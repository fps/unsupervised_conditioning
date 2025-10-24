import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics

dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls()

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Setting up latent")

#=
latent = (0.1 .+ t) .* 5f-1 .* cat(
    sin.(2 .* pi .* t),
    sin.(2 .* pi .* t .+ pi/4),
    sin.(2 .* pi .* t .+ pi/2),
    sin.(2 .* pi .* t .+ (3*pi/4)),
    dims=2) |> dev
=#

#=
latent = (0.1 .+ t) .* 5f-1 .* cat(
    sin.(2 .* pi .* t),
    sin.(2 .* pi .* t .+ pi/2),
    dims=2)
=#

# latent = 0.5 .* t

latent = 0.5 .* cat(
    sin.(1 .* 2 .* pi .* t),
    sin.(1 .* 2 .* pi .* t .+ pi/2),
    sin.(2 .* 2 .* pi .* t),
    sin.(2 .* 2 .* pi .* t .+ pi/2),
    sin.(4 .* 2 .* pi .* t),
    sin.(4 .* 2 .* pi .* t .+ pi/2),
    sin.(8 .* 2 .* pi .* t),
    sin.(8 .* 2 .* pi .* t .+ pi/2),
    sin.(16 .* 2 .* pi .* t),
    sin.(16 .* 2 .* pi .* t .+ pi/2),
    sin.(32 .* 2 .* pi .* t),
    sin.(32 .* 2 .* pi .* t .+ pi/2),
    sin.(64 .* 2 .* pi .* t),
    sin.(64 .* 2 .* pi .* t .+ pi/2),
    sin.(128 .* 2 .* pi .* t),
    sin.(128 .* 2 .* pi .* t .+ pi/2),
    sin.(256 .* 2 .* pi .* t),
    sin.(256 .* 2 .* pi .* t .+ pi/2),
    #=
    sin.(10 .* 2 .* pi .* t),
    sin.(10 .* 2 .* pi .* t .+ pi/2),
    sin.(11 .* 2 .* pi .* t),
    sin.(11 .* 2 .* pi .* t .+ pi/2),
    sin.(12 .* 2 .* pi .* t),
    sin.(12 .* 2 .* pi .* t .+ pi/2),
    sin.(13 .* 2 .* pi .* t),
    sin.(13 .* 2 .* pi .* t .+ pi/2),
    sin.(14 .* 2 .* pi .* t),
    sin.(14 .* 2 .* pi .* t .+ pi/2),
    sin.(15 .* 2 .* pi .* t),
    sin.(15 .* 2 .* pi .* t .+ pi/2),
    sin.(16 .* 2 .* pi .* t),
    sin.(16 .* 2 .* pi .* t .+ pi/2),
    =#
    dims=2
)

latent = repeat(Float32.(latent), 1, 1, 1) |> dev

num_latent = size(latent, 2)

println("Setting up model...")

channels = 24
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:10]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(2f-4), m_amp)

losses = []

println("Training...")
for n in 1:50000
    loss, grads = Flux.withgradient(m_amp) do m_amp
        Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent, x_dev, dims=2)))
    end

    push!(losses, loss)

    println("n: $n, loss: $loss")
    Flux.update!(opt, m_amp, grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(m_amp(cat(latent, x_dev, dims=2))[:,1,1] |> Flux.cpu),
            Plots.plot(y_dev[offset:end,1,1] |> Flux.cpu),
            Plots.plot(log10.(losses)),
            layout=(3,1),
            linewidth=0.1
        ) |> display
    end

    #=
    if n % 100 == 0
        Plots.plot(Plots.plot(y), Plots.plot(Flux.cpu(latent[:,:,1])), Plots.plot(controls), layout=(3,1), linewidth=0.1, legend=:none) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2))))")
    end
    =#
end
