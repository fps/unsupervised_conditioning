ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "4GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics

Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls1()

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)

x .-= x_mean
x ./= x_std

y .-= y_mean
y ./= y_std

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

x2, y2, fs2, sigmas2, controls2, t2 = synthetic_data_3controls1(5)

x2 .-= x_mean
x2 ./= x_std

y2 .-= y_mean
y2 ./= y_std

x2_dev = x2 |> dev
y2_dev = repeat(y2, 1, 1, 1) |> dev

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
    #=
    sin.(256 .* 2 .* pi .* t),
    sin.(256 .* 2 .* pi .* t .+ pi/2),
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

# latent = 0.5 .* t

latent = repeat(Float32.(latent), 1, 1, 1) |> dev
latent2 = repeat(latent, inner=(5,1,1), outer=(1,1,1))
num_latent = size(latent, 2)

println("Setting up model...")

channels = 24
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:5]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(2f-4), m_amp)

losses = []
losses_val = []

println("Training...")
for n in 1:50000
    loss, grads = Flux.withgradient(m_amp) do m_amp
        Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent, x_dev, dims=2)))
    end

    push!(losses, loss)

    push!(losses_val, Flux.mse(y2_dev[offset:end, :, :], m_amp(cat(latent2, x2_dev, dims=2))))

    # println("n: $n, loss: $loss")
    Flux.update!(opt, m_amp, grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(m_amp(cat(latent, x_dev, dims=2))[:,1,1] |> Flux.cpu, linewidth=0.1, label="Predicted"),
            Plots.plot(y_dev[offset:end,1,1] |> Flux.cpu, linewidth=0.1, label="Original"),
            # Plots.plot(latent[:,:,1] |> cpu),
            Plots.plot(20 .* log10.(losses), label="Training loss (dB)"),
            Plots.plot(20 .* log10.(losses_val), label="Validation loss (dB)"),
            layout=(4,1)
        ) |> display
    end
end
