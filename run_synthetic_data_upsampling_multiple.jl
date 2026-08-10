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
    fixed_latent_indices = Random.rand(1:length(x), 15)
    fixed_latent_values = hcat(controls...)[fixed_latent_indices, :] |> dev

    println("Setting up latent")
    num_latent = 3

    # l_latent = Int(ceil((48000+48000) * (1/2)^n_sub))

    l_latent = [0, 0, 0]
    l_latent[1] = Int(ceil(48000/2^12)) * 2
    l_latent[2] = l_latent[1] * 2
    l_latent[3] = l_latent[1] * 16
    
    latent = [1f-15 .* randn(Float32, l, 1, 1) for l in l_latent] |> dev

    println("Setting up model...")

    upsample = Flux.Upsample(2, :bilinear)
    m_latent = Flux.Chain(
        x -> [upsample(x[1]), upsample(x[2]), x[3]],
        x -> [upsample(x[1]), upsample(x[2]), x[3]],
        x -> [upsample(x[1]), upsample(x[2]), x[3]],
        x -> [upsample(x[1]), x[2], x[3]],
        x -> cat(x..., dims=2),
        [upsample for n in 1:8]...) |> dev
    

    # m_latent = Flux.Upsample(2^7, :bilinear) |> dev
    # m_latent = Flux.Upsample(2^7, :nearest) |> dev
    
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
end

println("Training...")
for n in 1:150000
# n = 0
# while true
    global losses
    global val_losses
    # global  n
    global loss

    limit = 1f-1
    
    loss, grads = Flux.withgradient([latent, m_amp]) do (latent, m_amp, )
        latent_smooth = m_latent(latent)[((1:48000) .+ 24000), :, :]

        l_mse = (length(train_mask) / sum(train_mask)) * Flux.mse(
            train_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            train_mask[offset:end,:,:] .* m_amp(cat(latent_smooth, x_dev, dims=2)))

        # l_mse += Flux.mse(latent_smooth[fixed_latent_indices, :, :], fixed_latent_values)

        l_limit = sum((latent_smooth .- min.(max.(latent_smooth, -limit), limit)).^2)
        # l_mse += 1f-0 * w_shift * Flux.mse(latent, circshift(latent, (shift, 0, 0)))

        l_mean  = 1f-0 * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)
        
        l_val = (length(val_mask) / sum(val_mask)) * Flux.mse(
            val_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            val_mask[offset:end,:,:] .* m_amp(cat(latent_smooth, x_dev, dims=2)))

        (l_mse + l_limit + l_mean, l_mse, l_val)
    end
    
    Flux.update!(opt, [latent, m_amp], grads[1])

    push!(losses, loss[1])
    push!(losses_mse, loss[2])
    push!(val_losses, loss[end])

    println("n: $n, losses: $(loss)")
    
    if n % 100 == 0
        Plots.plot(
            Plots.plot((1:length(y)) .+ offset, y, label="y"),
            Plots.plot(Flux.cpu(m_latent(latent)[((1:48000) .+ 24000), :]), label="latent_smooth"),
            Plots.plot(controls, label="controls"),
            Plots.plot(log10.(losses), label="training loss"),
            Plots.plot(log10.(losses_mse), label="mse"),
            # Plots.plot(log10.(losses_diff)),
            # Plots.plot(log10.(losses_var)),
            Plots.plot(log10.(val_losses), label="validation loss"),
            layout=(6,1)
        ) |> display
    end
    n += 1
end
