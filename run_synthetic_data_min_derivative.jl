ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, OhMyREPL

Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")

println("Generating synthetic data...")
fs = 48000
t = 0:(1/fs):(1-(1/fs))
x, y, controls = synthetic_data_3controls1()

x .-= Statistics.mean(x, dims=1)
x ./= Statistics.std(x, dims=1)

y .-= Statistics.mean(y, dims=1)
y ./= Statistics.std(y, dims=1)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev



Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Setting up latent")
num_latent = 3

latent = 1f-10 .* randn(Float32, length(x), num_latent, 1) |> dev

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

weights = 2 .^ ([(n-1) for n in 1:num_latent] ./ num_latent)'
weights_dev = weights |> dev
println("Training...")
for n in 1:150000
    noise = 1f-1 .* #= weights .* =# randn(Float32, size(latent)...) |> dev
    # noise = 1f-1 .* (1 .+ 10 .* [(n-1) for n in 1:num_latent] ./ num_latent)' .* randn(Float32, size(latent)...) |> dev
    (loss, loss_deriv, loss_recon, loss_recon_clean), grads = Flux.withgradient([latent, m_amp]) do params
        latent = params[1]
        m_amp = params[2]
        l_recon = 1f0 * Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent .+ noise, x_dev, dims=2)))
        l_recon_clean = Flux.mse(y_dev[offset:end, :, :], m_amp(cat(latent, x_dev, dims=2)))
        l_deriv = 1f3 * Statistics.mean(weights_dev .* diff(latent, dims=1).^2)
        l_mean = 1f0 * Statistics.mean(Statistics.mean(latent, dims=1).^2)
        # l_mag = 1f-1 * Statistics.mean(latent.^2)

        # l += 1f5 * Statistics.mean(diff(diff(latent, dims=1), dims=1).^2)
        # l += 1f7 * Statistics.mean(diff(diff(diff(latent, dims=1), dims=1), dims=1).^2)

        # l += 1f4 .* Statistics.mean((2 .* latent .- latent).^2)
        # l += (1f5*Statistics.mean(latent.^2) - Statistics.mean(latent.^2))^2
        # l += 1f0 .* Statistics.mean((I .- cov(latent)).^2)
        # l += 1f0 .* (1f0 - Statistics.mean(latent.^2)).^2

        (l_recon + l_deriv + l_mean, l_deriv, l_recon, l_recon_clean)
    end
    println("n: $n, losses: $loss, $loss_deriv, $loss_recon, $loss_recon_clean")
    Flux.update!(opt, [latent, m_amp], grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(y),
            Plots.plot((m_amp(cat(latent, x_dev, dims=2))|>Flux.cpu)[:,:,1]),
            [Plots.plot(Flux.cpu(cat(#= latent[:,k,1] .+ noise[:,k,1], =# latent[:,k,1], dims=2))) for k in 1:num_latent]...,
            Plots.plot(controls), layout=(3+num_latent,1),
            linewidth=0.2, legend=:none
        ) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        # println("losses: $loss, $loss_recon, $loss_recon_clean")
    end
end
