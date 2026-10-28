ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "4GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ParameterSchedulers, FFTW

dev = Flux.gpu
cpu = Flux.cpu

include("synthetic_data.jl")
include("util.jl")
include("half_band_filters.jl")
include("cat2.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls1()

# Random.seed!(2)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev


# Validation data
x2, y2, fs2, sigmas2, controls2, t2 = synthetic_data_3controls2()

# Random.seed!(2)

x2_dev = x2 |> dev
y2_dev = repeat(y2, 1, 1, 1) |> dev


Plots.default(dpi=300, linewidth=0.5, alpha=0.75, legend=:none, size=(1000,1000))

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1)) |> display

println("Setting up latent")
num_latent = 3


println("Setting up model...")

depth_encoder = 10
depth_decoder = 5

channels_encoder = 32
channels_decoder = 32
# activation = Flux.rrelu
# activation = Flux.leakyrelu
activation = Flux.tanh

m_encoder = Flux.Chain(
    Flux.Conv((1, ), 2=>channels_encoder),
    # [Flux.Chain(Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k), Flux.Dropout(0.5)) for k in 0:(depth_encoder-1)]...,
    # [Flux.Chain(Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k), Flux.Dropout(0.5; dims=2)) for k in 0:(depth_encoder-1)]...,
    [Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k) for k in 0:(depth_encoder-1)]...,
    Flux.Conv((1,), channels_encoder=>num_latent)
) |> dev

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels_decoder),
    [Flux.Conv((3,), channels_decoder=>channels_decoder, activation, dilation=2^k) for k in 0:(depth_decoder-1)]...,
    Flux.Conv((1,), channels_decoder=>1)
) |> dev

offset_encoder = 48000 - size(randn(Float32, 48000, 2, 1) |> dev |> m_encoder, 1)
# offset_latent = 48000 - size(randn(Float32, 48000, 3, 1) |> dev |> m_latent_down |> m_latent_up, 1)
offset_decoder = 48000 - size(randn(Float32, 48000, 1+num_latent, 1) |> dev |> m_decoder, 1)

lr = 2e-4
# warmup_epochs = 5000
max_epochs = 50000
schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(lr, 1e-5, 50000))


println("Setting up optimizer...")
# opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_latent_down, m_latent_up, m_decoder])
opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_decoder])
# opt = Flux.setup(Flux.AdamW(;eta=lr, lambda=0.1), [m_encoder, m_decoder])

I = sqrt(1f-0) .* eye(num_latent)
Im = ones(Float32, num_latent, num_latent) - I

loss_min = 1f10
losses_train = []
losses_val = []
lrs = []

N_x = size(x_dev, 1)
# w = Float32.(1 ./ (1:div(N_x, 2))) |> dev
# w = 1f0 .* Float32.((1:div(N_x,2)) ./ div(N_x,2)) |> dev

cutoff = 200

w = ones(Float32, div(N_x, 2) + 1) |> Flux.gpu
w[1:1] .= 1
w[2:cutoff] .= 0
w[(cutoff+1):end] .= 1

function loss(m_encoder, m_decoder, x, y, x2, y2)
    l = 0
    N_x = size(x, 1)

    encoded = cat(x, y, dims=2) |> m_encoder
    N_encoded = size(encoded, 1)

    encoded_fft = abs.(FFTW.fft(encoded, 1)[1:(div(N_x, 2)+1), :, :] ./ sqrt(N_x)).^2

    N_encoded_fft = size(encoded_fft, 1)

    # println("$(typeof(encoded_fft))")

    l += Statistics.mean(Statistics.mean(w .* encoded_fft, dims=1).^2)
    # l += Statistics.mean((1 .- (Statistics.mean((1 .- w) .* encoded_fft, dims=1))).^2)

    l += (1 - Statistics.mean(encoded.^2)).^2

    # decoded = cat2_centered(encoded, x2) |> m_decoder
    decoded = cat(encoded, x2[(div(N_x - N_encoded, 2) + 1):(div(N_x - N_encoded, 2)+N_encoded), :, :], dims=2) |> m_decoder

    N_decoded = size(decoded, 1)

    # decoded = cat2_causal(latent_upsampled, x) |> m_decoder


    # N4 = div(N_x - N_encoded, 2)

    l_recon = Flux.mse(decoded, y2[(1 + N_x - N_decoded):end, :, :])

    l += l_recon
    # l += l_recon

    # latent_smooth = m_encoder(cat(x, y, dims=2)) |> m_latent_down

    # latent_smooth += dev((1f1 * lr) .* randn(Float32, size(latent_smooth)...))

    # Derivative term
    # l += 1f-0 * Statistics.mean(Statistics.mean(diff(latent_downsampled, dims=1).^2, dims=1))

    # latent_and_input = cat(m_latent_up(latent_smooth), x[(1+offset_enc):end,:,:], dims=2)
    # latent_and_input = cat(m_latent_up(latent_smooth), x[(1+(offset_encoder + div(offset_latent, 2))):(end-div(offset_latent, 2)),:,:], dims=2)
    # latent_and_input = cat(m_latent_up(latent_smooth), x[1:(end-offset_enc),:,:], dims=2)

    # l += Flux.mse(y[(1+offset_encdec):end, :, :], m_decoder(latent_and_input)[:,:,:])

    # Overall mean term
    # l += 1f-0 * Statistics.mean(Statistics.mean(latent_downsampled.^2, dims=1))

    # Covariance term
    # l += 1f-0 * Statistics.mean((I .- Im .* cov(latent_upsampled)).^2)

    # Mean per latent term
    # l += 1f-0 * Statistics.mean(Statistics.mean(encoded, dims=1).^2)
    (l, l_recon)
end

println("Training...")
for n in 1:max_epochs
    global lr
    global loss_min
    global losses
    # loss, grads = Flux.withgradient([m_encoder, m_latent_down, m_latent_up, m_decoder]) do params
    losses, grads = Flux.withgradient([m_encoder, m_decoder]) do params
        m_encoder = params[1]
        m_decoder = params[2]
        loss(m_encoder, m_decoder, x_dev, y_dev, x_dev, y_dev)
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

    push!(losses_val, loss(m_encoder, m_decoder, x2_dev, y2_dev, x2_dev, y2_dev)[2])

    if n % 100 == 0
        Plots.plot(
            Plots.plot((m_encoder(cat(x_dev, y_dev, dims=2)) |> Flux.cpu)[:,:,1]),
            Plots.plot((m_encoder(cat(x2_dev, y2_dev, dims=2)) |> Flux.cpu)[:,:,1]),
            # Plots.plot((m_encoder(cat(x_dev, y_dev, dims=2)) |> m_latent_down |> Flux.cpu)[:,:,1]),
            Plots.plot((lrs)),
            Plots.plot(log10.(losses_train)),
            Plots.plot(log10.(losses_val)),
            layout=(5,1),
            size=(1000, 1000)
        ) |> display
    end
    # CUDA.reclaim()
end
