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
fs = 12000

x, y, controls = synthetic_data_3controls1(1, fs)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

# coeff = Float32.(DSP.digitalfilter(DSP.Lowpass(10), DSP.FIRWindow(DSP.Windows.hamming(ord)), fs=fs))

println("Setting up latent")
num_latent = 3

# latent = 1f-1 .* randn(Float32, length(x)+(ord-1), num_latent, 1) |> dev
# latent = 1f-0 .* randn(Float32, length(x), num_latent, 1) |> dev

latent = cat(
    Flux.ConvTranspose(ones(Float32, 128, 1, 1))(sum(DSP.mt_spectrogram(y, 128, 127).power[10:40,:], dims=1)'[:,:,:])[:,1,1],
    Flux.ConvTranspose(ones(Float32, 128, 1, 1))(sum(DSP.mt_spectrogram(y, 128, 127).power[1:10,:], dims=1)'[:,:,:])[:,1,1],
    Flux.ConvTranspose(ones(Float32, 128, 1, 1))(sum(DSP.mt_spectrogram(y, 128, 127).power[50:end,:], dims=1)'[:,:,:])[:,1,1],
    dims=2
    ) |> dev

latent ./= Flux.ConvTranspose(ones(Float32, 128, 1, 1))(ones(Float32, fs-127, 1, 1))[:,1,1] |> dev
latent .-= Statistics.mean(latent, dims=1)
latent ./= Statistics.std(latent, dims=1)
latent = latent[:,:,:]

println("Fixing some latents...")
fixed_latent_indices = Random.rand(1:length(x), 75)
fixed_latent_values = hcat(controls...)[fixed_latent_indices, :] |> dev
fixed_latent_values .-= Statistics.mean(fixed_latent_values, dims=1)

println("Setting up model...")
m_latent = Flux.Conv(repeat(coeff, 1, 1, num_latent), groups=num_latent) |> dev

channels = 16
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:4]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + fs - size(m_amp(randn(Float32, fs, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(1f-4), [latent, m_amp])
Flux.adjust!(opt[1], 1f-3)
I = sqrt(1f-2) .* eye(num_latent)


losses = []
diff_weights = Float32.(0.1.^(0:(num_latent-1)))' |> dev
losses_mse = []
val_losses = []
println("Training...")
n = 0
while true
    global n
# for n in 1:50000
    #=
    if n % 1000 == 0
        global latent = 1f-10 .* randn(Float32, length(x)+(ord-1), num_latent, 1) |> dev
    end
    =#
    r_latent = 1f-1 .* randn(Float32, size(latent)...) |> dev
    r_latent_smooth = 1f-2 .* randn(Float32, size(x_dev, 1), num_latent, 1) |> dev
    loss, grads = Flux.withgradient([latent, m_amp]) do params
        # latent_smooth = m_latent(params[1] .+ r_latent)
        # latent_smooth = m_latent(params[1]) .+ r_latent_smooth
        l_mse = Flux.mse(y_dev[offset:end, :, :], params[2](cat(params[1], x_dev, dims=2)))

        # l += #= 10.0^k =# #= (1/n) =# 1f-5 * sum(latent_smooth.^2) / prod(size(latent_smooth))
        # l += Flux.mse(latent_smooth[fixed_latent_indices, :, :], fixed_latent_values)
        # l += 1f-0 * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)
        # l += 1f-0 * Statistics.mean(latent_smooth.^2)
        # l += 1f-0 * Statistics.mean((I .- cov(latent_smooth)).^2)
        l_mean = 0f-0 * Statistics.mean(params[1].^2)
        l_diff = 1f-0 * Float32(fs) * Statistics.mean(diff_weights .* diff(params[1], dims=1).^2)

        l_var = 1f-0 * Statistics.mean(
            (1f-0 .- Statistics.var(params[1], dims=1)).^2
        )

        (l_mse + l_mean + l_diff + l_var, l_mse)
    end
    Flux.update!(opt, [latent, m_amp], grads[1])

    push!(losses, loss[1])
    push!(losses_mse, loss[2])



    if n % 100 == 0
        @info "n: $n, losses: $loss"
        # Plots.plot(Plots.plot(y), Plots.plot(Flux.cpu(m_latent(latent))[:,:,1]), Plots.plot(controls), layout=(3,1), linewidth=0.1, legend=:none) |> display
        Plots.plot(
            Plots.plot(y),
            Plots.plot(Flux.cpu(latent)[:,:,1]),
            Plots.plot(controls),
            Plots.plot(log10.(losses)),
            Plots.plot(log10.(losses_mse)),
            layout=(5,1),
            linewidth=0.1,
            legend=:none
        ) |> display
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2))))")
    end
    n += 1
end
