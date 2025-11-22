ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ProgressBars

Plots.unicodeplots()
Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu
cpu = Flux.cpu

include("util.jl")
include("chunks.jl")
include("time_encodings.jl")

println("Reading training data")

# y, y_fs = WAV.wavread("data/fender_blues_junior/Take7_Audio 1-1.3.wav");
y, y_fs = WAV.wavread("data/fender_blues_junior/Take6_Audio 1-1.wav");

y = Float32.(y)

#y = y[1:(div(length(y), 2))]
y = y[1:(48000*60*3)]

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

t = Float32.(0:(1/length(x)):(1-(1/length(x))))

println("Setting up latent")

# latent = cat(t .- 0.5, fourier_basis(t, 6), dims=2)
latent = Float32.(cat(t .- 0.5, exponential_fourier_basis(t, 7), dims=2))
# latent = t .- 0.5

# latent = 0.5 .* t

latent = repeat(Float32.(latent), 1, 1, 1)
num_latent = size(latent, 2)

println("Setting up model...")

channels = 32
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    # [Flux.SkipConnection(Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k), (mx, x) -> mx .+ 1f-0 .* x[(1+size(x,1)-size(mx,1)):end,:,:]) for k in 0:11]...,
    [ Flux.SkipConnection(Flux.Chain([Flux.Conv((2,), channels=>channels, Flux.leakyrelu, dilation=2^k) for k in 0:8]...), (mx, x) -> mx .+ x[(1+size(x,1)-size(mx,1)):end,:,:]) for l in 1:6 ]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(2f-4), m_amp)

losses = []
losses_val = []

chunksize = 2*offset
batchsize=div(48000*2, chunksize)

println("offset: $offset, chunksize: $chunksize, batchsize: $batchsize")

x_chunks, y_chunks, chunk_starts = chunks(cat(x, latent, dims=2), y, chunksize, offset)

N_chunks = size(x_chunks, 3)
N_val_chunks = div(N_chunks, 10)

println("N_chunks: $N_chunks, N_val_chunks: $N_val_chunks")

shuffle = Random.randperm(N_chunks)

x_chunks = x_chunks[:,:,shuffle]
y_chunks = y_chunks[:,:,shuffle]


x_val = x_chunks[:,:,1:N_val_chunks] |> dev
y_val = y_chunks[:,:,1:N_val_chunks] |> dev

x_chunks = x_chunks[:,:,(N_val_chunks+1):end]
y_chunks = y_chunks[:,:,(N_val_chunks+1):end]

println("Training...")
for n in 0:50000

    losses_epoch = []

    data_loader = Flux.MLUtils.DataLoader((x_chunks, y_chunks), batchsize=batchsize, shuffle=true, parallel=true, partial=true)

    for batch in ProgressBars.ProgressBar(data_loader, printing_delay=0.1)
        batch_dev = batch |> dev

        loss, grads = Flux.withgradient(m_amp) do m_amp
            Flux.Losses.mse(batch_dev[2], m_amp(batch_dev[1]))
        end

        CUDA.unsafe_free!(batch_dev[1])
        CUDA.unsafe_free!(batch_dev[2])

        push!(losses_epoch, loss)

        # println("n: $n, loss: $loss")
        Flux.update!(opt, m_amp, grads[1])
    end

    push!(losses, Statistics.mean(losses_epoch))
    push!(losses_val, Flux.mse(y_val, m_amp(x_val)))
    # push!(losses_val, Flux.mse(y_dev[(start_index + offset):end_index, :, :], m_amp(cat(latent[start_index:end_index, :, :], x_dev[start_index:end_index, :, :], dims=2))))
      if n % 1 == 0
        Plots.plot(
            # Plots.plot(m_amp(cat(latent, x_dev, dims=2))[:,1,1] |> Flux.cpu, linewidth=0.1, label="Predicted"),
            # Plots.plot(y_dev[offset:end,1,1] |> Flux.cpu, linewidth=0.1, label="Original"),
            # Plots.plot(latent[:,:,1] |> cpu),
            Plots.plot(log10.(losses), label="Training loss (log10)"),
            Plots.plot(log10.(losses_val), label="Validation loss (log10)"),
            # Plots.plot(m_amp(x_val[:,:,1:1])[:,1,1] |> cpu, label="Validation output"),
            # Plots.plot(y_val[:,1,1] |> cpu, label="Validation target"),
            # Plots.plot(m_amp(x_chunks[:,:,1:1] |> dev)[:,1,1] |> cpu, label="Training output"),
            # Plots.plot(y_chunks[:,1,1], label="Training target"),
            layout=(2,1),
        ) |> display
    end
end
