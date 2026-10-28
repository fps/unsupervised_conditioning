if !@isdefined(init) || init
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

    import Pkg

    println("Activating package...")
    Pkg.activate(".")

    println("Importing packages...")
    import CUDA, Flux, WAV, Plots, DSP, Statistics, FFTW

    Plots.GR.reset()
    
    Plots.default(size=(1024, 1280), linewidth=0.1, dpi=300)

    dev = Flux.gpu
    cpu = Flux.cpu

    include("synthetic_data.jl")
    include("util.jl")

    println("Generating synthetic data...")
    fs = 48000

    x, y, controls = synthetic_data_3controls1(1)

    x .-= Statistics.mean(x)
    x ./= Statistics.std(x)

    y .-= Statistics.mean(y)
    y ./= Statistics.std(y)

    x_dev = x |> dev
    y_dev = repeat(y, 1, 1, 1) |> dev

    for n in 1:length(controls)
        # @info "n: $n"
        controls[n] = collect(controls[n])
        controls[n] .-= Statistics.mean(controls[n])
        controls[n] ./= maximum(abs.(controls[n]))
    end

    Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1)) |> display

    println("Fixing some latents...")
    fixed_latent_indices = Random.rand(1:length(x), 25)
    fixed_latent_values = hcat(controls...)[fixed_latent_indices, :] |> dev
    # fixed_latent_values .-= Statistics.mean(fixed_latent_values, dims=1)
    # fixed_latent_values ./= Statistics.std(fixed_latent_values, dims=1)

    println("Setting up latent")
    num_latent = 8

    n_sub = 8

    l_latent = Int(ceil((48000+48000) * (1/2)^n_sub))

    latent = 1f-15 .* randn(Float32, l_latent, num_latent, 1) |> dev

    println("Setting up model...")
    latent_coeff = [25f0, 25f0]; latent_coeff ./= (5f-1 * sum(latent_coeff))

    m_latent = Flux.Chain([
        Flux.Chain(
            Flux.ConvTranspose(Float32.(repeat([1, 1], 1, 1, num_latent)), stride=2, groups=num_latent),
            Flux.Conv(Float32.(repeat(DSP.digitalfilter(DSP.Lowpass(0.5), DSP.FIRWindow(DSP.hamming(15))), 1, 1, num_latent)), groups=num_latent))
        for n in 1:n_sub]) |> dev

    latent_offset = div(size(m_latent(latent), 1) - 48000, 2)
    # latent_offset = size(m_latent(latent), 1) - 48000

    channels = 16
    m_amp = Flux.Chain(
        Flux.Conv((1, ), (1+num_latent)=>channels),
        [Flux.Conv((3,), channels=>channels, Flux.leakyrelu, dilation=2^k) for k in 0:7]...,
        Flux.Conv((1,), channels=>1)
    ) |> dev

    offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

    println("Setting up optimizer...")
    # opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
    opt = Flux.setup(Flux.Adam(5f-4), [latent, m_amp])

    I = sqrt(1f0) .* eye(num_latent)

    losses = []
    losses_mse = []
    val_losses = []

    train_mask = ones(Float32, 48000, 1, 1) |> dev
    train_mask[24000:24100] .= 0

    val_mask = 1f0 .- train_mask

    diff_weights = Float32.(0.1.^(0:(num_latent-1))') |> dev

    f2 = DSP.digitalfilter(DSP.Lowpass(100/48000), DSP.FIRWindow(DSP.hanning(10000)))
end

println("Training...")
for n in 1:20000
# n = 0
# while true
    global losses
    global val_losses
    # global  n
    global loss

    loss, grads = Flux.withgradient([latent, m_amp]) do (latent, m_amp, )
        latent_smooth = m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:]
        # latent_smooth = m_latent(latent)[1:48000,:,:]

        l_mse = (length(train_mask) / sum(train_mask)) * Flux.mse(
            train_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            train_mask[offset:end,:,:] .* m_amp(cat(latent_smooth, x_dev, dims=2)))

        # l_mse += Flux.mse(latent_smooth[fixed_latent_indices, :, :], fixed_latent_values)
        # l_mse += Flux.mse(latent_smooth, min.(max.(latent_smooth, -1), 1))

        l_mean = 0f-0 * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)
        l_weight = 0f1 * Statistics.mean(abs.(latent_smooth))

        l_diff = 0f0 .* Float32(fs) * Statistics.mean(diff_weights .* diff(latent_smooth, dims=1).^2)

        l_var = 0f0 # 1f-0 * Statistics.mean((1f-0 .- Statistics.var(latent_smooth, dims=1)).^2)

        l_cov = 0f0 * Statistics.mean((I .- cov(latent_smooth)).^2)

        l_val = (length(val_mask) / sum(val_mask)) * Flux.mse(
            val_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            val_mask[offset:end,:,:] .* m_amp(cat(latent_smooth, x_dev, dims=2)))

        (
            l_mse + l_mean + l_weight + l_diff + l_var + l_cov,
            l_mse, l_mean, l_weight, l_diff, l_var, l_cov, l_val
        )
    end
    Flux.update!(opt, [latent, m_amp], grads[1])

    # latent .*= (1f0 - 1f-2)

    push!(losses, loss[1])
    push!(losses_mse, loss[2])
    push!(val_losses, loss[end])

    println("n: $n, losses: $(loss)")

    # Flux.update!(opt, latent2, grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(y, label="y"),
            Plots.plot(Flux.cpu(m_latent(latent)[latent_offset .+ (1:48000),:,1]), label="latent_smooth"),
            Plots.plot(Flux.cpu(latent[:,:,1]), label="latent"),
            Plots.plot(controls, label="controls"),
            Plots.plot(log10.(losses), label="training loss"),
            Plots.plot(log10.(losses_mse), label="mse"),
            # Plots.plot(log10.(losses_diff)),
            # Plots.plot(log10.(losses_var)),
            Plots.plot(log10.(val_losses), label="validation loss"),
            layout=(7,1)
        ) |> display

        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:], x_dev, dims=2))))")
    end
    n += 1
end

@info "Gathering spectral data..."
spectrum_hop = 80

latent_smooth = m_latent(latent)[latent_offset .+ (1:48000),:,1]

spectra_starts = 1:spectrum_hop:(48000 - ((2 * n_spectrum_bins) + offset + spectrum_hop))

n_spectrum_samples = 256

n_spectrum_bins = 64

spectra = []


for n in spectra_starts
    @info "n: $n"
    spectrum_input_sample = dev(randn(Float32, (2*n_spectrum_bins)+offset, 1, n_spectrum_samples))    # current_latent = repeat(latent_smooth[n:n, :, :], (2*n_spectrum_bins)+offset, 1, n_spectrum_samples)
    current_latent = repeat(latent_smooth[n:(n+(2*n_spectrum_bins)+offset-1),:,:], 1, 1, n_spectrum_samples)
    y_hat = m_amp(cat(current_latent, spectrum_input_sample, dims=2))
    spectrum = Statistics.mean(cat([abs.(FFTW.fft(y_hat[:,1,n])) for n in 1:n_spectrum_samples]..., dims=2), dims=2)
    spectrum = spectrum[1:(div(size(spectrum, 1), 2)), :]
    push!(spectra, spectrum)
end

spectrogram = cpu(cat(spectra..., dims=2))

Plots.heatmap(log10.(spectrogram)) |> display

#=
ica = MultivariateStats.fit(MultivariateStats.ICA, log10.(spectrogram), 3)
Plots.plot(MultivariateStats.predict(ica, log10.(spectrogram))')
=#

pca = MultivariateStats.fit(MultivariateStats.PCA, log10.(spectrogram), maxoutdim=3)
Plots.plot(MultivariateStats.predict(pca, log10.(spectrogram))', linewidth=1)

