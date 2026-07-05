ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

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

println("Reading training data...")

y, y_fs = WAV.wavread("data/fender_blues_junior/Take7_Audio 1-1.3.wav");
# y, y_fs = WAV.wavread("data/fender_blues_junior/Take6_Audio 1-1.wav");

y = Float32.(y)

#y = y[1:(div(length(y), 2))]
y = y[1:(48000*60*1)]

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)

y .-= y_mean
y ./= y_std

x, x_fs = WAV.wavread("data/fender_blues_junior/varying_noise_20_min_seed_0.wav");

x = Float32.(x)

t = Float32.(0:(1/length(x)):(1-(1/length(x))))

x = x[1:length(y)]
t = t[1:length(y)]

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)

x .-= x_mean
x ./= x_std


println("Setting up latent...")

# latent = cat(t .- 0.5, fourier_basis(t, 6), dims=2)
latent = Float32.(cat(t .- 0.5, exponential_fourier_basis(t, 8), dims=2))
# latent = t
# latent = zeros(Float32, length(y))
# latent = []

# latent = t .- 0.5

# latent = 0.5 .* t

latent = repeat(Float32.(latent), 1, 1, 1)
num_latent = size(latent, 2)
# num_latent = 0

println("Setting up model...")

additive(mx, x) = mx .+ Float32(1/3) .* x[(1+size(x,1)-size(mx,1)):end,:,:]

channels = 16
m_amp = Flux.Chain(
  Flux.Conv((1,), (1+num_latent) => channels),
  [ Flux.Conv((3,), channels => channels, Flux.tanh; dilation=2^k) for k in 0:9]...,
  [ Flux.Conv((3,), channels => channels, Flux.tanh; dilation=2^k) for k in 0:9]...,
  Flux.Conv((1,), channels => 1)) |> dev

#=
channels = 24
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    # [Flux.SkipConnection(Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k), (mx, x) -> mx .+ x[(1+size(x,1)-size(mx,1)):end,:,:]) for k in 0:11]...,
    [ Flux.SkipConnection(Flux.Chain([Flux.Conv((max(3, 8-k),), channels=>channels, Flux.leakyrelu, groups=1, dilation=2^k) for k in 0:8]...), additive) for l in 1:4 ]...,
		# [ Flux.SkipConnection(Flux.Conv((3,), channels=>channels, Flux.leakyrelu, dilation=2^k), additive) for k in 0:10 ]...,
		# [ Flux.SkipConnection(Flux.Conv((3,), channels=>channels, Flux.leakyrelu, dilation=2^k), additive) for k in 0:10 ]...,
    # [ Flux.SkipFlux.Chain([Flux.Conv((3,), channels=>channels, Flux.leakyrelu, dilation=2^k) for k in 0:11]..., Flux.BatchNorm(channels)), (mx, x) -> mx .+ x[(1+size(x,1)-size(mx,1)):end,:,:]) for l in 1:2 ]...,
    Flux.Conv((1,), channels=>1)
) |> dev
=#

# m_amp = AmpModeling.NAM.Wavenet.Standard(1+num_latent, 1, 16, 8, false, false) |> dev

offset = 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

losses = []
losses_val = []

chunksize = 2*offset
batchsize=div(48000*1, chunksize)

println("offset: $offset, chunksize: $chunksize, batchsize: $batchsize")

x_chunks, y_chunks, chunk_starts = chunks(cat(x, latent, dims=2), y, chunksize, offset)
# x_chunks, y_chunks, chunk_starts = chunks(x, y, chunksize, offset)

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

lr = 1e-3 * sqrt(batchsize)
println("Setting up optimizer: $(lr)...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(lr), m_amp)
#opt = Flux.setup(Flux.Momentum(lr, m_amp)

n_warmup_epochs = 10

lr_max = 2e-4

lrs = range(1e-8, lr_max, n_warmup_epochs)

n_epochs = 50000

lrs = vcat(lrs, lr_max * ones(n_epochs - n_warmup_epochs))

println("Training...")
for n in 1:n_epochs
    Flux.adjust!(opt, lrs[n])

    losses_epoch = []

    data_loader = Flux.MLUtils.DataLoader((x_chunks, y_chunks), batchsize=batchsize, shuffle=true, parallel=false, partial=true)

    for batch in ProgressBars.ProgressBar(data_loader, printing_delay=0.5)
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

    losses_epoch_val = []

    data_loader_val = Flux.MLUtils.DataLoader((x_val, y_val), batchsize=batchsize, shuffle=true, parallel=false, partial=true)

    for batch in ProgressBars.ProgressBar(data_loader_val, printing_delay=0.5)
        batch_dev = batch |> dev

        loss = Flux.Losses.mse(batch_dev[2], m_amp(batch_dev[1]))

        CUDA.unsafe_free!(batch_dev[1])
        CUDA.unsafe_free!(batch_dev[2])

        push!(losses_epoch_val, loss)

        # println("n: $n, loss: $loss")
    end

    push!(losses_val, Statistics.mean(losses_epoch_val))
    # push!(losses_val, Flux.mse(y_dev[(start_index + offset):end_index, :, :], m_amp(cat(latent[start_index:end_index, :, :], x_dev[start_index:end_index, :, :], dims=2))))
    if n % 1 == 0
        UnicodePlots.lineplot(log10.(losses), title="Training loss (log10)", width=:auto) |> display
        UnicodePlots.lineplot(log10.(losses_val), title="Validation loss (log10)", width=:auto) |> display
    end
end
