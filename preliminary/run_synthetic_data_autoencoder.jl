ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random


dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls()

# Random.seed!(2)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Designing lowpass filter...")
#=
ord = DSP.remezord(100/fs, 150/fs, 0.1, 0.001)
coeff = Float32.(DSP.digitalfilter(DSP.Lowpass(60), DSP.FIRWindow(DSP.Windows.hann(ord)), fs=fs))
=#

#=
ord = 4000
coeff = DSP.remez(ord, [(0, 100) => 1, (150, 24000) => 0], Hz=48000)
=#

ord = 1000
coeff = Float32.(DSP.digitalfilter(DSP.Lowpass(10), DSP.FIRWindow(DSP.Windows.hann(ord)), fs=fs))

println("Setting up latent")
num_latent = 3

# latent = 1f0 .* randn(Float32, length(x)+(ord-1), num_latent, 1) |> dev

#=
println("Fixing some latents...")
fixed_latent_indices = Random.rand(1:length(x), 200)
fixed_latent_values = hcat(controls...)[fixed_latent_indices, :] |> dev
fixed_latent_values .-= Statistics.mean(fixed_latent_values, dims=1)
=#

println("Setting up model...")
m_latent = Flux.Conv(repeat(coeff, 1, 1, num_latent), groups=num_latent) |> dev

channels = 16

m_encoder = Flux.Chain(
    Flux.Conv((1, ), 1=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
        Flux.Conv((1,), channels=>num_latent)
) |> dev

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
        Flux.Conv((1,), channels=>1)
) |> dev

offset = 48000 - size(m_decoder(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
opt = Flux.setup(Flux.Adam(1f-3), [m_encoder, m_decoder])
# opt = Flux.setup(Flux.AdamW(eta=2f-4, lambda=0.00001), [m_encoder, m_decoder])

I = sqrt(1f-2) .* eye(num_latent)

println("Training...")
for n in 1:50000
    loss, grads = Flux.withgradient([m_encoder, m_decoder]) do params
        m_encoder = params[1]
        m_decoder = params[2]

        latent = m_encoder(y_dev) |> m_latent

        latent_and_input = cat(latent, x_dev[(ord+offset):end,:,:], dims=2)

        l = Flux.mse(y_dev[ord+(2*offset):end, :, :], m_decoder(latent_and_input))

        l += 1f-2 * Statistics.mean(latent.^2)
        # l += 1f-3 * Statistics.mean((I .- cov(latent)).^2)
        l += 1f-2 * Statistics.mean(Statistics.mean(latent, dims=1).^2)
        l
    end
    println("n: $n, loss: $loss")
    Flux.update!(opt, [m_encoder, m_decoder], grads[1])

    if n % 100 == 0
        Plots.plot((m_encoder(y_dev) |> m_latent |> Flux.cpu)[:,:,1], linewidth=0.1) |> display
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
