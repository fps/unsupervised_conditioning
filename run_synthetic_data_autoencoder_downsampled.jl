import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ParameterSchedulers


dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")
include("half_band_filters.jl")
include("cat2.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls()

# Random.seed!(2)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.default(dpi=300, linewidth=0.5, alpha=0.75, legend=:none)

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1)) |> display

println("Designing lowpass filter...")
ord = DSP.remezord(100/fs, 150/fs, 0.1, 0.001)
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
#=
m_latent_down = Flux.Conv(ones(480, 1, num_latent) ./ (4*480), stride=120, groups=num_latent) |> dev
m_latent_up = Flux.ConvTranspose(ones(480, 1, num_latent), groups=num_latent, stride=120) |> dev
=#

#=
m_latent_down = Flux.Conv(repeat(Float32.(DSP.Windows.hanning(480)) ./ 240, 1, 1, 3), stride=240, groups=num_latent) |> dev
m_latent_up = Flux.ConvTranspose(repeat(Float32.(DSP.Windows.hanning(480)), 1, 1, 3), groups=num_latent, stride=240) |> dev
=#

# #=

stages = 8
taps = 15
cutoff = 0.4

m_latent_down = downsampler(stages, taps, num_latent, cutoff) |> dev
m_latent_up = upsampler(stages, taps, num_latent, cutoff) |> dev

# =#

#=
m_latent_down = Flux.Conv((480,), num_latent => num_latent, stride=240, groups=num_latent) |> dev
m_latent_up = Flux.ConvTranspose((480,), num_latent => num_latent, stride=240, groups=num_latent) |> dev
=#

depth_encoder = 8
depth_decoder = 5

channels_encoder = 16
channels_decoder = 16
# activation = Flux.rrelu
# activation = Flux.leakyrelu
activation = Flux.tanh

m_encoder = Flux.Chain(
    Flux.Conv((1, ), 2=>channels_encoder),
    [Flux.Conv((3,), channels_encoder=>channels_encoder, activation, dilation=2^k) for k in 0:(depth_encoder-1)]...,
    Flux.Conv((1,), channels_encoder=>num_latent)
) |> dev

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels_decoder),
    [Flux.Conv((3,), channels_decoder=>channels_decoder, activation, dilation=2^k) for k in 0:(depth_decoder-1)]...,
    Flux.Conv((1,), channels_decoder=>1)
) |> dev

offset_encoder = 48000 - size(randn(Float32, 48000, 2, 1) |> dev |> m_encoder, 1)
offset_latent = 48000 - size(randn(Float32, 48000, 3, 1) |> dev |> m_latent_down |> m_latent_up, 1)
offset_decoder = 48000 - size(randn(Float32, 48000, 4, 1) |> dev |> m_decoder, 1)

#=
offset_test_data = randn(Float32, 48000, 2, 1) |> dev

offset_test_encoded_data = offset_test_data |> m_encoder |> m_latent_down |> m_latent_up

offset_enc = 48000 - size(offset_test_encoded_data, 1)

offset_test_input_data = randn(Float32, 48000 - offset_enc) |> dev

offset_test_encoded_data2 = cat(offset_test_encoded_data, offset_test_input_data, dims=2)

offset_test_decoded_data = offset_test_encoded_data2 |> m_decoder

offset_encdec = 48000 - size(offset_test_decoded_data, 1)
=#

lr = 1e-3
schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(lr, 1e-4, 5000))


println("Setting up optimizer...")
# opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_latent_down, m_latent_up, m_decoder])
opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_decoder])
# opt = Flux.setup(Flux.AdamW(;eta=lr, lambda=0.1), [m_encoder, m_decoder])

I = sqrt(1f-0) .* eye(num_latent)
Im = ones(Float32, num_latent, num_latent) - I

loss_min = 1f10
losses = []

println("Training...")
for n in 1:50000
    global lr
    global loss_min
    global losses
    # loss, grads = Flux.withgradient([m_encoder, m_latent_down, m_latent_up, m_decoder]) do params
    loss, grads = Flux.withgradient([m_encoder, m_decoder]) do params
        N_x_dev = size(x_dev, 1)

        global lr
        l = 0
        m_encoder = params[1]
        m_decoder = params[2]

        encoded = cat(x_dev, y_dev, dims=2) |> m_encoder
        N_encoded = size(encoded, 1)

        latent_downsampled = encoded |> m_latent_down
        latent_upsampled = latent_downsampled |> m_latent_up
        N_latent_upsampled = size(latent_upsampled, 1)

        # latent_upsampled_size = size(latent_upsampled, 1)

        decoded = cat2_centered(latent_upsampled, x_dev) |> m_decoder
        N_decoded = size(decoded, 1)

        # decoded = cat2_causal(latent_upsampled, x_dev) |> m_decoder


        N4 = div(N_x_dev - N_latent_upsampled, 2)

        l_recon = Flux.mse(decoded, y_dev[(1+N4+offset_decoder):(N4+offset_decoder+N_decoded), :, :])

        l += l_recon
        # l += l_recon

        # latent_smooth = m_encoder(cat(x_dev, y_dev, dims=2)) |> m_latent_down

        # latent_smooth += dev((1f1 * lr) .* randn(Float32, size(latent_smooth)...))

        # Derivative term
        # l += 1f-0 * Statistics.mean(Statistics.mean(diff(latent_downsampled, dims=1).^2, dims=1))

        # latent_and_input = cat(m_latent_up(latent_smooth), x_dev[(1+offset_enc):end,:,:], dims=2)
        # latent_and_input = cat(m_latent_up(latent_smooth), x_dev[(1+(offset_encoder + div(offset_latent, 2))):(end-div(offset_latent, 2)),:,:], dims=2)
        # latent_and_input = cat(m_latent_up(latent_smooth), x_dev[1:(end-offset_enc),:,:], dims=2)

        # l += Flux.mse(y_dev[(1+offset_encdec):end, :, :], m_decoder(latent_and_input)[:,:,:])

        # Overall mean term
        # l += 1f-0 * Statistics.mean(Statistics.mean(latent_downsampled.^2, dims=1))

        # Covariance term
        # l += 1f-0 * Statistics.mean((I .- Im .* cov(latent_upsampled)).^2)

        # Mean per latent term
        # l += 1f-0 * Statistics.mean(Statistics.mean(latent_downsampled, dims=1).^2)
        (l, l_recon)
    end

    if loss[1] < loss_min
        global m_min = deepcopy([m_encoder, m_decoder])
        loss_min = loss[1]
    end

    global lr = ParameterSchedulers.next!(schedule)
    # Flux.adjust!(opt, lr + lr * abs(randn()))
    Flux.adjust!(opt, eta = lr + 1f-4 * abs(randn()))
    # Flux.adjust!(opt, lr)
    # println("n: $n, lr: $lr, loss: $loss, reconstruction loss: $l_recon")
    println("n: $n, lr: $lr, loss: $(loss[1]), reconstruction loss: $(loss[2])")
    # Flux.update!(opt, [m_encoder, m_latent_down, m_latent_up, m_decoder], grads[1])
    Flux.update!(opt, [m_encoder, m_decoder], grads[1])

    push!(losses, loss[1])
    if n % 100 == 0
        Plots.plot(
            Plots.plot((m_encoder(cat(x_dev, y_dev, dims=2)) |> m_latent_down |> m_latent_up |> Flux.cpu)[:,:,1]),
            Plots.plot((m_encoder(cat(x_dev, y_dev, dims=2)) |> m_latent_down |> Flux.cpu)[:,:,1]),
            Plots.plot(log10.(losses)),
            layout=(3,1),
            # size=(600, 1200)
        ) |> display
    end
    # CUDA.reclaim()
end
