ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics

dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")

println("Generating synthetic data...")
x, y, controls = synthetic_data_3controls1()
fs = 48000

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Designing lowpass filter...")
ord = DSP.remezord(100/fs, 150/fs, 0.1, 0.001)

coeff = Float32.(DSP.digitalfilter(DSP.Lowpass(10), DSP.FIRWindow(DSP.Windows.hamming(ord)), fs=fs))

println("Setting up latent")
num_latent = 3

latent = 1f-1 .* randn(Float32, length(x)+(ord-1), num_latent, 1) |> dev

println("Fixing some latents...")
fixed_latent_indices = Random.rand(1:length(x), 75)
fixed_latent_values = hcat(controls...)[fixed_latent_indices, :] |> dev
fixed_latent_values .-= Statistics.mean(fixed_latent_values, dims=1)

println("Setting up model...")
m_latent = Flux.Conv(repeat(coeff, 1, 1, num_latent), groups=num_latent) |> dev

channels = 16
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(2f-4), [latent, m_amp])

I = sqrt(1f-2) .* eye(num_latent)

w_d = 1f-4

println("Training...")
for n in 1:50000
    #=
    if n % 1000 == 0
        global latent = 1f-10 .* randn(Float32, length(x)+(ord-1), num_latent, 1) |> dev
    end
    =#
    r_latent = 1f-1 .* randn(Float32, size(latent)...) |> dev
    r_latent_smooth = 1f-2 .* randn(Float32, size(x_dev, 1), num_latent, 1) |> dev

    idx1 = Random.rand(1:size(x_dev, 1))
    idx2 = Random.rand(1:size(x_dev, 1))

    loss, grads = Flux.withgradient([latent, m_amp]) do params
        # latent_smooth = m_latent(params[1] .+ r_latent)
        latent_smooth = m_latent(params[1]) .+ r_latent_smooth
        l = Flux.mse(y_dev[offset:end, :, :], params[2](cat(latent_smooth, x_dev, dims=2)))

        # l += #= 10.0^k =# #= (1/n) =# 1f-5 * sum(latent_smooth.^2) / prod(size(latent_smooth))
        # l += Flux.mse(latent_smooth[fixed_latent_indices, :, :], fixed_latent_values)
        l += 1f-0 * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)
        # l += 1f-0 * Statistics.mean(latent_smooth.^2)
        # l += 1f-0 * Statistics.mean((I .- cov(latent_smooth)).^2)

        # l += 1f-2 * max(0f0, (5f-1 - LinearAlgebra.norm(latent_smooth[idx1, :, :] .- latent_smooth[idx2, :, :])).^2 - 1f-1)
        l += w_d * max(0f0, (5f-1 - LinearAlgebra.norm(latent_smooth[idx1, 1, :] .- latent_smooth[idx1, 2, :])).^2 - 1f-1)
        l += w_d * max(0f0, (5f-1 - LinearAlgebra.norm(latent_smooth[idx1, 2, :] .- latent_smooth[idx1, 3, :])).^2 - 1f-1)
        l += w_d * max(0f0, (5f-1 - LinearAlgebra.norm(latent_smooth[idx1, 3, :] .- latent_smooth[idx1, 1, :])).^2 - 1f-1)

        l
    end
    println("n: $n, loss: $loss")
    Flux.update!(opt, [latent, m_amp], grads[1])

    if n % 100 == 0
        Plots.plot(Plots.plot(y), Plots.plot(Flux.cpu(m_latent(latent))[:,:,1]), Plots.plot(controls), layout=(3,1), linewidth=0.1, legend=:none) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2))))")
    end
end
