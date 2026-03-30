ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "4GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ParameterSchedulers, MultivariateStats, ShiftedArrays

dev = Flux.gpu
cpu = Flux.cpu

include("synthetic_data.jl")
include("util.jl")
include("half_band_filters.jl")
include("cat2.jl")

println("Generating synthetic data...")
fs = 48000
x, y, controls = synthetic_data_3controls1()
x_std = sqrt(Statistics.var(x))
y_std = sqrt(Statistics.var(y))

#=
x = x
y = y
=#

x ./= x_std
y ./= y_std

println("x_std: $(x_std), y_std: $(y_std)")

# Random.seed!(2)

x_dev = x |> dev
y_dev = y |> dev

# Validation data
x2, y2, controls2 = synthetic_data_3controls2()

x2 ./= x_std
y2 ./= y_std

# Random.seed!(2)

x2_dev = x2 |> dev
y2_dev = y2 |> dev

# Validation data
x3, y3, controls3 = synthetic_data_3controls1()

x3 ./= x_std
y3 ./= y_std
# Random.seed!(2)

x3_dev = x3 |> dev
y3_dev = y3 |> dev


Plots.default(dpi=300, linewidth=0.5, alpha=0.75, legend=:none, size=(1000,1800))

Plots.plot(Plots.plot(x[:,1,1]), Plots.plot(y[:,1,1]), Plots.plot(controls), layout=(3,1)) |> display

println("Designing lowpass filter...")
ord = DSP.remezord(100/fs, 150/fs, 0.1, 0.001)
coeff = Float32.(DSP.digitalfilter(DSP.Lowpass(10), DSP.FIRWindow(DSP.Windows.hann(ord)), fs=fs))

println("Setting up latent")
num_latent = 3


println("Setting up model...")

max_shift = 1000

depth_encoder = 12
depth_decoder = 5

channels_encoder = 32
channels_decoder = 32
# activation = Flux.rrelu
# activation = Flux.leakyrelu
activation = Flux.tanh

m_encoder = Flux.Chain(
    # Flux.Dropout(0.5),
    Flux.Conv((1, ), 2=>channels_encoder),
    # [Flux.Chain(Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k), Flux.Dropout(0.6 * (1.0 - (k+1)/(depth_encoder+2)))) for k in 0:(depth_encoder-1)]...,
    # [Flux.Chain(Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k), Flux.Dropout(0.8^(k+1))) for k in 0:(depth_encoder-1)]...,
    # [Flux.Chain(Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k), Flux.Dropout(0.5; dims=2)) for k in 0:(depth_encoder-1)]...,
    [Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k) for k in 0:(depth_encoder-1)]...,
    # [Flux.WeightNorm(Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k), :weight) for k in 0:(depth_encoder-1)]...,
    Flux.Conv((1,), channels_encoder=>num_latent)
) |> dev

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels_decoder),
    [Flux.Conv((3,), channels_decoder=>channels_decoder, activation, dilation=2^k) for k in 0:(depth_decoder-1)]...,
    Flux.Conv((1,), channels_decoder=>1)
) |> dev

offset_encoder = 48000 - size(randn(Float32, 48000, 2, 1) |> dev |> m_encoder, 1)
offset_decoder = 48000 - size(randn(Float32, 48000, 1+num_latent, 1) |> dev |> m_decoder, 1)

lr = 1e-3
# warmup_epochs = 5000
max_epochs = 50000
schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(lr, 1e-4, 50000))


println("Setting up optimizer...")
# opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_latent_down, m_latent_up, m_decoder])
opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_decoder])
# opt = Flux.setup(Flux.AdamW(;eta=lr, lambda=0.1), [m_encoder, m_decoder])

I = sqrt(1f-0) .* eye(num_latent)
Im = ones(Float32, num_latent, num_latent) - I

loss_min = 1f10
losses_train = []
losses_val1 = []
losses_val2 = []
lrs = []

function loss(m_encoder, m_decoder, x, y, x2, y2)
    l = 0
    N_x = size(x, 1)

    encoded = cat(x[:,:,:], y[:,:,:], dims=2) |> m_encoder

    # encoded = circshift(encoded, (sum(Random.rand([-1,1], 10000)), 0, 0))

    N_encoded = size(encoded, 1)

    decoded = cat2_centered(encoded, x2[:,:,:]) |> m_decoder
    N_decoded = size(decoded, 1)

    N4 = div(N_x - N_encoded, 2)

    l_recon = Flux.mse(decoded, y2[(1+N4+offset_decoder):(N4+offset_decoder+N_decoded), :, :])

    # l_recon = Flux.mse(decoded, y2[(1 + N_x - N_decoded):end])

    l += l_recon

    # Mean per latent term
    l += 1f-0 * Statistics.mean(Statistics.mean(encoded, dims=1).^2)
    (l, l_recon)
end

println("Training...")
for n in 1:max_epochs
    global lr
    global loss_min
    global losses
    #=
    x_shifted = copy(ShiftedArrays.ShiftedArray(x, (sum(Random.rand([-1,1], max_shift)), 0, 0), default = 0)) .+ 1f0 .* randn(Float32, size(x_dev)...) |> dev
    y_shifted = copy(ShiftedArrays.ShiftedArray(y, (sum(Random.rand([-1,1], max_shift)), 0, 0), default = 0)) .+ 1f0 .* randn(Float32, size(y_dev)...) |> dev
    =#

    # #=
    x_shifted = x_dev # .+ 1f-0 .* dev(randn(Float32, size(x_dev)...))
    y_shifted = y_dev # .+ 1f-0 .* dev(randn(Float32, size(y_dev)...))
    # =#

    losses, grads = Flux.withgradient([m_encoder, m_decoder]) do params
        m_encoder = params[1]
        m_decoder = params[2]

        loss(m_encoder, m_decoder, x_shifted, y_shifted, x_dev, y_dev)
    end

    if losses[1] < loss_min
        global m_min = deepcopy([m_encoder, m_decoder])
        loss_min = losses[1]
    end

    #=
    if n < warmup_epochs
        # Flux.adjust!(opt, lr * (1 - (1 - n/warmup_epochs)))
        effective_lr = lr * (1 - (1 - n/warmup_epochs))
    else
        # Flux.adjust!(opt, lr * (1 - (n-warmup_epochs)/(max_epochs-warmup_epochs)))
        effective_lr = lr * (1 - (n-warmup_epochs)/(max_epochs-warmup_epochs))
    end
    Flux.adjust!(opt, effective_lr)
    push!(lrs, effective_lr)
    =#
    global lr = ParameterSchedulers.next!(schedule)
    push!(lrs, lr)
    # Flux.adjust!(opt, lr + lr * abs(randn()))
    Flux.adjust!(opt, eta = lr + 1f-4 * abs(randn()))
    # Flux.adjust!(opt, lr)

    # println("n: $n, lr: $lr, loss: $loss, reconstruction loss: $l_recon")
    println("n: $n, lr: $lr, loss: $(losses[1]), reconstruction loss: $(losses[2])")
    # Flux.update!(opt, [m_encoder, m_latent_down, m_latent_up, m_decoder], grads[1])
    Flux.update!(opt, [m_encoder, m_decoder], grads[1])

    push!(losses_train, losses[2])

    push!(losses_val1, loss(m_encoder, m_decoder, x2_dev, y2_dev, x2_dev, y2_dev)[2])

    push!(losses_val2, loss(m_encoder, m_decoder, x_dev, y_dev, x3_dev, y3_dev)[2])

    if n % 100 == 0
        latent = m_encoder(cat(x_dev, y_dev, dims=2)[:,:,:]) |> cpu

        pca = MultivariateStats.fit(MultivariateStats.PCA, latent[:,:,1]')

        Plots.plot(
            Plots.plot((m_encoder(cat(x_dev, y_dev, dims=2)[:,:,:]) |> Flux.cpu)[:,:,1]),
            Plots.plot((m_encoder(cat(x2_dev, y2_dev, dims=2)[:,:,:]) |> Flux.cpu)[:,:,1]),
            # Plots.plot((m_encoder(cat(x_dev, y_dev, dims=2)) |> m_latent_down |> Flux.cpu)[:,:,1]),
            Plots.plot((lrs)),
            Plots.plot((pca.proj' * latent[:,:,1]')'),
            Plots.plot(20 .* log10.(losses_train)),
            Plots.plot(20 .* log10.(losses_val1)),
            Plots.plot(20 .* log10.(losses_val2)),
            layout=(7,1)
            ) |> display
    end
    # CUDA.reclaim()
end
