import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random

Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls1()

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Setting up latent")
num_latent = 3

latent = 1f-10 .* randn(Float32, length(x) + 10000, num_latent, 1) |> dev

println("Setting up model...")

channels = 16
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(1f-4), [latent, m_amp])

I = 1f0 .* eye(3)

println("Training...")
for n in 1:150000
    shift = sum(Random.rand([-1, 0, 1], 10000))
    # shift = Random.rand(-1000:1000)
    loss, grads = Flux.withgradient([latent, m_amp]) do params
        latent = params[1]
        m_amp = params[2]
        # l = 1f0 * Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent[1:length(x), :, :], x_dev, dims=2)))
        l = 1f0 * Flux.mse(y_dev[offset:end, :, :], m_amp(cat(circshift(latent, (shift, 0, 0))[1:length(x), :, :], x_dev, dims=2)))
        # l += 1f0 * Statistics.mean((latent .- circshift(latent, (shift, 0, 0))).^2)

        # l += 1f4 .* Statistics.mean((2 .* latent .- latent).^2)
        # l += (1f5*Statistics.mean(latent.^2) - Statistics.mean(latent.^2))^2
        # l += 1f0 .* Statistics.mean((I .- cov(latent)).^2)
        # l += 1f0 .* (1f0 - Statistics.mean(latent.^2)).^2

        l
    end
    println("n: $n, loss: $loss")
    Flux.update!(opt, [latent, m_amp], grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(y),
            [Plots.plot(Flux.cpu(latent)[:,k,1]) for k in 1:num_latent]...,
            Plots.plot(controls), layout=(2+num_latent,1),
            linewidth=0.2, legend=:none
        ) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent[1:length(x),:,:], x_dev, dims=2))))")
    end
end
