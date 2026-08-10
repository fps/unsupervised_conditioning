if !@isdefined(init) || init
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

    import Pkg

    println("Activating package...")
    Pkg.activate(".")

    println("Importing packages...")
    import CUDA, Flux, WAV, Plots, DSP, Statistics

    Plots.GR.reset()
    
    Plots.default(size=(1024, 1280), linewidth=0.1, dpi=300)

    dev = Flux.gpu
    cpu = Flux.cpu

    include("synthetic_data.jl")
    include("util.jl")
    include("time_encodings.jl")

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
    num_latent = 3

    # t = range(-1, 1, 48000)[:,:,:] |> dev
    t = exponential_fourier_basis(range(0, 1-(1/48000), 48000), 8)[:,:,:] |> dev
    
    m_latent = Flux.Chain(
        Flux.Conv((1,), size(t, 2)  => 8 * num_latent),
        Flux.Conv((1,), 8 * num_latent => 8 * num_latent, Flux.leakyrelu),
        Flux.Conv((1,), 8 * num_latent => 8 * num_latent, Flux.leakyrelu),
        Flux.Conv((1,), 8 * num_latent => 4 * num_latent, Flux.leakyrelu),
        Flux.Conv((1,), 4 * num_latent => 2 * num_latent, Flux.leakyrelu),
        Flux.Conv((1,), 2 * num_latent => num_latent)) |> dev
    
    channels = 16
    m_amp = Flux.Chain(
        Flux.Conv((1, ), (1+num_latent)=>channels),
        [Flux.Conv((3,), channels=>channels, Flux.leakyrelu, dilation=2^k) for k in 0:7]...,
        Flux.Conv((1,), channels=>1)
    ) |> dev

    offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

    lr = 5f-4
    println("Setting up optimizer...")

    opt = Flux.setup(Flux.Adam(lr), [m_latent, m_amp])
    # opt = Flux.setup(Flux.AdamW(; eta = lr, lambda = 0.1), [m_latent, m_amp])

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
for n in 1:150000
# n = 0
# while true
    global losses
    global val_losses
    # global  n
    global loss

    loss, grads = Flux.withgradient([m_latent, m_amp]) do (m_latent, m_amp, )
        # latent_smooth = m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:]
        latent_smooth = m_latent(t)
        # latent_smooth = m_latent(latent)[1:48000,:,:]

        l_mse = (length(train_mask) / sum(train_mask)) * Flux.mse(
            train_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            train_mask[offset:end,:,:] .* m_amp(cat(latent_smooth, x_dev, dims=2)))

        l_mse += Flux.mse(latent_smooth[fixed_latent_indices, :, :], fixed_latent_values)
        l_mse += Flux.mse(latent_smooth, min.(max.(latent_smooth, -1), 1))

        l_mean = 1f-0 * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)
        l_weight = 0f1 * Statistics.mean(latent_smooth.^2)

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
    Flux.update!(opt, [m_latent, m_amp], grads[1])

    # latent .*= (1f0 - 1f-2)

    push!(losses, loss[1])
    push!(losses_mse, loss[2])
    push!(val_losses, loss[end])

    println("n: $n, losses: $(loss)")

    # Flux.update!(opt, latent2, grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(y, label="y"),
            Plots.plot(Flux.cpu(m_latent(t)[:,:,1]), label="latent_smooth"),
            Plots.plot(controls, label="controls"),
            Plots.plot(log10.(losses), label="training loss"),
            Plots.plot(log10.(losses_mse), label="mse"),
            # Plots.plot(log10.(losses_diff)),
            # Plots.plot(log10.(losses_var)),
            Plots.plot(log10.(val_losses), label="validation loss"),
            layout=(6,1)
        ) |> display

        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))")
        # println("losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:], x_dev, dims=2))))")
    end
    n += 1
end
