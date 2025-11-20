ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "6GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random

Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu

include("util.jl")

println("Reading training data")

y, y_fs = WAV.wavread("data/fender_blues_junior/Take7_Audio 1-1.3.wav");

y = Float32.(y)

y = y[1:(div(length(y), 100))]

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)

y .-= y_mean
y ./= y_std

x, x_fs = WAV.wavread("data/fender_blues_junior/varying_noise_20_min_seed_0.wav");

x = Float32.(x)

x = x[1:length(y)]

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)

x .-= x_mean
x ./= x_std

x_dev = x |> dev
y_dev = y |> dev

t = Float32.(0:(1/length(x)):(1-(1/length(x))))

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

fpi = Float32(pi)

latent = 5f-1 .* cat(
    sin.(1 .* 2 .* fpi .* t),
    sin.(1 .* 2 .* fpi .* t .+ fpi/2),
    sin.(2 .* 2 .* fpi .* t),
    sin.(2 .* 2 .* fpi .* t .+ fpi/2),
    sin.(4 .* 2 .* fpi .* t),
    sin.(4 .* 2 .* fpi .* t .+ fpi/2),
    sin.(8 .* 2 .* fpi .* t),
    sin.(8 .* 2 .* fpi .* t .+ fpi/2),
    #=
    sin.(16 .* 2 .* fpi .* t),
    sin.(16 .* 2 .* fpi .* t .+ fpi/2),
    sin.(32 .* 2 .* fpi .* t),
    sin.(32 .* 2 .* fpi .* t .+ fpi/2),
    sin.(64 .* 2 .* fpi .* t),
    sin.(64 .* 2 .* fpi .* t .+ fpi/2),
    sin.(128 .* 2 .* fpi .* t),
    sin.(128 .* 2 .* fpi .* t .+ fpi/2),
    sin.(256 .* 2 .* fpi .* t),
    sin.(256 .* 2 .* fpi .* t .+ fpi/2),

    sin.(10 .* 2 .* fpi .* t),
    sin.(10 .* 2 .* fpi .* t .+ fpi/2),
    sin.(11 .* 2 .* fpi .* t),
    sin.(11 .* 2 .* fpi .* t .+ fpi/2),
    sin.(12 .* 2 .* fpi .* t),
    sin.(12 .* 2 .* fpi .* t .+ fpi/2),
    sin.(13 .* 2 .* fpi .* t),
    sin.(13 .* 2 .* fpi .* t .+ fpi/2),
    sin.(14 .* 2 .* fpi .* t),
    sin.(14 .* 2 .* fpi .* t .+ fpi/2),
    sin.(15 .* 2 .* fpi .* t),
    sin.(15 .* 2 .* fpi .* t .+ fpi/2),
    sin.(16 .* 2 .* fpi .* t),
    sin.(16 .* 2 .* fpi .* t .+ fpi/2),
    =#
    dims=2
)

# latent = 0.5 .* t

latent = repeat(Float32.(latent), 1, 1, 1) |> dev
num_latent = size(latent, 2)

println("Setting up model...")

channels = 32
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:12]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(2f-4), m_amp)

losses = []
losses_val = []

chunksize = 48000
N_chunks = div(length(x), chunksize)


println("Training...")
for n in 1:50000

    losses_epoch = []

    p = Random.randperm(N_chunks)
    for chunk_index in p
        start_index = 1 + (chunksize * (chunk_index-1))
        end_index = start_index + chunksize - 1

        loss, grads = Flux.withgradient(m_amp) do m_amp
            Flux.mse(y_dev[(start_index + offset):end_index, :, :], m_amp(cat(latent[start_index:end_index, :, :], x_dev[start_index:end_index, :, :], dims=2)))
        end

        push!(losses_epoch, loss)

        println("n: $n, loss: $loss")
        Flux.update!(opt, m_amp, grads[1])
    end

    push!(losses, Statistics.mean(losses_epoch))

    if n % 100 == 1
        Plots.plot(
            # Plots.plot(m_amp(cat(latent, x_dev, dims=2))[:,1,1] |> Flux.cpu, linewidth=0.1, label="Predicted"),
            # Plots.plot(y_dev[offset:end,1,1] |> Flux.cpu, linewidth=0.1, label="Original"),
            # Plots.plot(latent[:,:,1] |> cpu),
            Plots.plot(20 .* log10.(losses), label="Training loss (dB)"),
            layout=(1,1)
        ) |> display
    end
end
