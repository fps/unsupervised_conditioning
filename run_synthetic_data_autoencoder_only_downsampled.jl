import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ParameterSchedulers


dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")
include("half_band_filters.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls()

# Random.seed!(2)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

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
cutoff = 0.5

m_latent_down = downsampler(stages, taps, num_latent, cutoff) |> dev
m_latent_up = upsampler(stages, taps, num_latent, cutoff) |> dev

# =#

#=
m_latent_down = Flux.Conv((480,), num_latent => num_latent, stride=240, groups=num_latent) |> dev
m_latent_up = Flux.ConvTranspose((480,), num_latent => num_latent, stride=240, groups=num_latent) |> dev
=#

channels = 16

m_encoder = Flux.Chain(
    Flux.Conv((1, ), 1=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
    Flux.Conv((1,), channels=>num_latent)
) |> dev

m_decoder = Flux.Chain(
    Flux.Conv((1, ), num_latent=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset_test_data = randn(Float32, 48000, 1, 1) |> dev

offset_test_encoded_data = offset_test_data |> m_encoder |> m_latent_down |> m_latent_up

offset_enc = 48000 - size(offset_test_encoded_data, 1)

offset_test_input_data = randn(Float32, 48000 - offset_enc) |> dev

# offset_test_encoded_data2 = cat(offset_test_encoded_data, offset_test_input_data, dims=2)

offset_test_decoded_data = offset_test_encoded_data |> m_decoder

offset_encdec = 48000 - size(offset_test_decoded_data, 1)

lr = 1e-3
schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(lr, 1e-4, 5000))


println("Setting up optimizer...")
# opt = Flux.setup(Flux.Adam(2f-4), [m_encoder, m_latent_down, m_latent_up, m_decoder])
opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_decoder])
# opt = Flux.setup(Flux.AdamW(eta=2f-4, lambda=0.00001), [m_encoder, m_decoder])

I = sqrt(1f-2) .* eye(num_latent)

loss_min = 1f10
losses = []

println("Training...")
for n in 1:50000
    global losses
    # loss, grads = Flux.withgradient([m_encoder, m_latent_down, m_latent_up, m_decoder]) do params
    loss, grads = Flux.withgradient([m_encoder, m_decoder]) do params
        l = 0
        m_encoder = params[1]
        m_decoder = params[2]

        latent_smooth = m_encoder(y_dev) |> m_latent_down

        # l += 1f-2 * Statistics.mean(latent_smooth.^2)
        # l += 1f-0 * Statistics.mean((I .- cov(latent_smooth)).^2)
        l += 1f-0 * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)

        # l += 1f-0 * Statistics.mean(Statistics.mean(diff(latent_smooth, dims=1).^2, dims=1))

        latent_and_input = m_latent_up(latent_smooth)

        l += Flux.mse(y_dev[(1+offset_encdec):end, :, :], m_decoder(latent_and_input)[:,:,:])

        l
    end

    Flux.adjust!(opt, ParameterSchedulers.next!(schedule))
    println("n: $n, lr: $lr, loss: $loss")
    # Flux.update!(opt, [m_encoder, m_latent_down, m_latent_up, m_decoder], grads[1])
    Flux.update!(opt, [m_encoder, m_decoder], grads[1])

    push!(losses, loss)
    if n % 100 == 0
        Plots.plot(
            Plots.plot((m_encoder(y_dev) |> m_latent_down |> m_latent_up |> Flux.cpu)[:,:,1]),
            Plots.plot(log10.(losses)),
            layout=(2,1),
            linewidth=0.2
        ) |> display
    end
    # CUDA.reclaim()
end
