ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "8GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ProgressBars, UnicodePlots

# Plots.unicodeplots()
Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu
cpu = Flux.cpu


include("../AmpModeling.jl/src/AmpModeling.jl")
include("util.jl")
include("chunks.jl")
include("time_encodings.jl")
include("synthetic_data.jl")

println("Reading training data...")

# #=
y, y_fs = WAV.wavread("data/fender_blues_junior/Take7_Audio 1-1.3.wav");
# y, y_fs = WAV.wavread("data/fender_blues_junior/Take6_Audio 1-1.wav");

y = Float32.(y)

# y = y[1:(div(length(y), 2))]
# y = y[1:(48000*60*1)]

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
# =#

# t = Float32.(0:(1/length(x)):(1-(1/length(x))))
t = Float32.(range(0, 1 - (1/length(x)), length(x)))

println("Setting up latent...")

latent_sampling_rate = 8
latent_period = Int(floor(y_fs / latent_sampling_rate))

num_latent = 8
latent = 1f-5 .* randn(Float32, Int(ceil(length(y) / latent_period)), num_latent, 1) |> dev

m_latent = Flux.Upsample(latent_period, :bilinear) |> dev

println("Setting up model...")

channels = 24
depth = 10
m_amp = Flux.Chain(
    Flux.Conv((1,), (1+num_latent) => channels),
    Flux.Chain([
        Flux.SkipConnection(
            Flux.Chain([ Flux.Conv((3,), channels => channels, Flux.leakyrelu; dilation=2^k) for k in 0:(depth-1)]...),
            (mx, x) -> (x[(1 + size(x, 1) - size(mx, 1)):end, :, :] .+ mx) ./ 2)
        for l in 1:2]...),
    Flux.Conv((1,), channels => 1)) |> dev

offset = 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

chunksize = Int(ceil(48000 / latent_period) * latent_period)
chunksize_latent = div(chunksize, latent_period)
chunk_overlap = Int(ceil(offset / latent_period)) * latent_period

chunk_hop = chunksize - chunk_overlap
chunk_hop_latent = div(chunk_hop, latent_period)

println("offset: $offset, chunksize: $chunksize, chunk_overlap: $chunk_overlap")

N_chunks = div(length(y), chunk_hop)

println("Setting up optimizer...")
n_warmup_epochs = 100

lr_min = 1e-8
lr_max = 1e-4 # * sqrt(batchsize)

opt = Flux.setup(Flux.Adam(lr_min), [m_amp, latent])

lrs = range(lr_min, lr_max, n_warmup_epochs)

n_epochs = 50000

lrs = vcat(lrs, lr_max * ones(n_epochs - n_warmup_epochs))

println("Training...")

losses = []
losses_val = []

parallel = true

x_dev = x |> dev
y_dev = y |> dev

for n in 1:n_epochs
    Flux.adjust!(opt, lrs[n])

    losses_epoch = []

    chunk_indices = Random.randperm(N_chunks)

    for index in ProgressBars.ProgressBar(chunk_indices)
        loss, grads = Flux.withgradient([m_amp, latent]) do (m_amp, latent)
            Flux.Losses.mse(y_dev[(1+offset+(index-1)*chunk_hop):(chunksize+(index-1)*chunk_hop), :, :], m_amp(cat(x_dev[(1+(index-1)*chunk_hop):(chunksize+(index-1)*chunk_hop), :, :], m_latent(latent[(1+(index-1)*chunk_hop_latent):(chunksize_latent+(index-1)*chunk_hop_latent), :, :]), dims=2)))
        end

        push!(losses_epoch, loss)

        # println("n: $n, loss: $loss")
        Flux.update!(opt, [m_amp, latent], grads[1])
    end

    push!(losses, Statistics.mean(losses_epoch))

    if (n-1) % 1 == 0
        for l in 1:num_latent
            UnicodePlots.lineplot(cpu(latent[1:div(size(latent, 1), 10), l, 1]), width=:auto) |> display
        end
        UnicodePlots.lineplot(log10.(losses), title="Training loss (log10)", width=:auto) |> display
    end
end
