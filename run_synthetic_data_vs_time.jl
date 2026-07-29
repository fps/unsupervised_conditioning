ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "4GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, FFTW

Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

Plots.GR.reset()

dev = Flux.gpu
cpu = Flux.cpu

include("synthetic_data.jl")
include("util.jl")
include("time_encodings.jl")

println("Generating synthetic data...")
x, y, controls = synthetic_data_3controls1()

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

x2, y2, controls2 = synthetic_data_3controls1(5)

x2 .-= x_mean
x2 ./= x_std

y2 .-= y_mean
y2 ./= y_std

x2_dev = x2 |> dev
y2_dev = repeat(y2, 1, 1, 1) |> dev

Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Setting up latent")

t = range(0, 1-(1/48000), 48000)
# latent = t[:,:,:] .- 0.5 |> dev
# f_latent = 50
latent = cat(t[:,:,:], exponential_fourier_basis(t, 8)[:,:,:], dims=2) |> dev
# latent = t .* cat(sin.(2 * pi * f_latent .* t), cos.(2 * pi * f_latent .* t), dims=2)[:,:,:] |> dev
# latent = 0.5 .* t

num_latent = size(latent, 2)

println("Setting up model...")

channels = 16
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.leakyrelu, dilation=2^k) for k in 0:7]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(5f-4), m_amp)

losses = []
losses_val = []

println("Training...")
for n in 1:50000
    loss, grads = Flux.withgradient(m_amp) do m_amp
        y = m_amp(cat(latent, x_dev, dims=2))
        y_hat = y_dev[offset:end, :, :]

        y_train = cat(y[1:24500,:,:], y[25001:end,:,:], dims=1)
        y_hat_train = cat(y_hat[1:24500,:,:], y_hat[25001:end,:,:], dims=1)
        
        l_train = Flux.mse(y_hat_train, y_train)

        y_test = y[24501:25000,:,:]
        y_hat_test = y_hat[24501:25000,:,:]

        l_test = Flux.mse(y_hat_test, y_test)

        (l_train, l_test)
    end

    push!(losses, loss[1])
    push!(losses_val, loss[2])

    # println("n: $n, loss: $loss")
    Flux.update!(opt, m_amp, grads[1])
    
    if n % 100 == 0
        Plots.plot(
            Plots.plot(m_amp(cat(latent, x_dev, dims=2))[:,1,1] |> Flux.cpu, linewidth=0.1, label="Predicted"),
            Plots.plot(y_dev[offset:end,1,1] |> Flux.cpu, linewidth=0.1, label="Original"),
            # Plots.plot(latent[:,:,1] |> cpu),
            Plots.plot(log10.(losses), label="Training loss (log10)"),
            Plots.plot(log10.(losses_val), label="Validation loss (log10)"),
            layout=(4,1)
        ) |> display
    end
end

@info "Feeding white noise samples and calculate spectrogram..."

spectra_starts = 1:10:size(t,1)

n_spectrum_samples = 50

spectra = []

for n in 1:10:size(t, 1)
    @info "n: $n"
    y_hat = m_amp(cat(repeat(latent[n:n, :, :], 2*offset, 1, n_spectrum_samples), dev(randn(Float32, 2*offset, 1, n_spectrum_samples)), dims=2))
    spectrum = Statistics.mean(cat([abs.(FFTW.fft(y_hat[:,1,n])) for n in 1:n_spectrum_samples]..., dims=2), dims=2)
    spectrum = spectrum[1:(div(size(spectrum, 1), 2)), :]
    push!(spectra, spectrum)
end

spectrogram = cat(spectra..., dims=2)

Plots.heatmap(cpu(log2.(spectrogram))) |> display



