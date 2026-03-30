ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

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
fs = 48000
x, y, controls = synthetic_data_3controls1()

# Random.seed!(2)

x_dev = x |> dev
y_dev = y |> dev

Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Setting up latent")
num_latent = 3


println("Setting up model...")
channels = 16

m_encoder = Flux.Chain(
    Flux.Conv((1, ), 2=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:10]...,
        Flux.Conv((1,), channels=>num_latent)
) |> dev

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
        Flux.Conv((1,), channels=>1)
) |> dev

offset_enc = 48000 - size(m_encoder(randn(Float32, 48000, 2, 1)|>dev), 1)
offset_dec = 48000 - size(m_decoder(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
opt = Flux.setup(Flux.Adam(1f-3), [m_encoder, m_decoder])
# opt = Flux.setup(Flux.AdamW(eta=2f-4, lambda=0.00001), [m_encoder, m_decoder])

I = sqrt(1f-2) .* eye(num_latent)

println("Training...")
for n in 1:50000
    # noise = 1f-1 .* randn(Float32, size(x_dev, 1)-offset_enc, num_latent, 1) |> dev
    x_noise = 1f-1 .* randn(Float32, size(x_dev)...) |> dev
    y_noise = 1f-1 .* randn(Float32, size(y_dev)...) |> dev

    loss, grads = Flux.withgradient([m_encoder, m_decoder]) do params
        m_encoder = params[1]
        m_decoder = params[2]

        latent = m_encoder(cat(x_dev .+ x_noise, y_dev .+ y_noise, dims=2)[:,:,:])
        # latent .+= 1f-2 .* dev(randn(Float32, size(latent)...))
        x_new = x_dev[(1+offset_enc):end,:,:]
        # x_with_noise = x_new .+ 1f-0 * dev(randn(Float32, size(x_new)...))
        latent_and_input = cat(latent #=.+ noise =#, x_new, dims=2)

        l = Flux.mse(y_dev[(1+offset_enc+offset_dec):end, :, :], m_decoder(latent_and_input))

        l += 1f0 * Statistics.mean(diff(latent, dims=1).^2)
        # l += 1f-2 * Statistics.mean(latent.^2)
        # l += 1f-3 * Statistics.mean((I .- cov(latent)).^2)
        l += 1f-0 * Statistics.mean(Statistics.mean(latent, dims=1).^2)
        l
    end
    println("n: $n, loss: $loss")
    Flux.update!(opt, [m_encoder, m_decoder], grads[1])

    if n % 100 == 0
        the_latent = (m_encoder(cat(x_dev, y_dev, dims=2)[:,:,:]) |> Flux.cpu)[:,:,1]
        Plots.plot([Plots.plot(the_latent[:,k]) for k in 1:size(the_latent, 2)]..., layout=(3,1), linewidth=0.1) |> display
    end
    #=
    if n % 100 == 0
        Plots.plot(Plots.plot(y), Plots.plot(Flux.cpu(m_latent(latent))[:,:,1]), Plots.plot(controls), layout=(3,1), linewidth=0.1, legend=:none) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2))))")
    end
    =#
    # CUDA.reclaim()
end
