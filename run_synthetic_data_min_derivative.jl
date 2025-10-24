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
num_latent = 3

latent = 1f-0 .* randn(Float32, length(x), num_latent, 1) |> dev

println("Setting up model...")

channels = 16
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 4, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(2f-4), [latent, m_amp])

I = 1f0 .* eye(3)

println("Training...")
for n in 1:150000
    loss, grads = Flux.withgradient([latent, m_amp]) do params
        latent = params[1]
        m_amp = params[2]
        l = Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent, x_dev, dims=2)))

        l += 1f0 .* Statistics.mean(diff(latent, dims=1).^2)

        l += 1f0 .* Statistics.mean((I .- cov(latent)).^2)

        l
    end
    println("n: $n, loss: $loss")
    Flux.update!(opt, [latent, m_amp], grads[1])

    if n % 100 == 0
        Plots.plot(Plots.plot(y), Plots.plot(Flux.cpu(latent)[:,:,1]), Plots.plot(controls), layout=(3,1), linewidth=0.2, legend=:none) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent, x_dev, dims=2))))")
    end
end
