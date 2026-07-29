if init
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

    import Pkg

    @info "Activating package..."
    Pkg.activate(".")

    @info "Importing packages..."
    import CUDA, Flux, WAV, Plots, DSP, Statistics

    Plots.default(size=(1024, 1280), linewidth=0.2, dpi=300, background_color_legend=Plots.Colors.RGBA(1, 1, 1, 0.5))

    dev = Flux.gpu
    cpu = Flux.cpu

    include("synthetic_data.jl")
    include("util.jl")
    include("losses.jl")

    @info "Generating synthetic data..."
    fs = 48000

    x, y, controls = synthetic_data_3controls1(1)
    x_dev = x |> dev
    y_dev = repeat(y, 1, 1, 1) |> dev

    Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1)) |> display


    @info "Initial latents"
    # #=
    windowsize=1024

    spec = Float32.(sqrt.(DSP.spectrogram(y, windowsize, windowsize-1, window=DSP.gaussian(windowsize, 0.1)).power))

    initial_latent = cat(
        Flux.ConvTranspose(ones(Float32, windowsize, 1, 1))(sum(spec[1:10,:], dims=1)'[:,:,:])[:,1,1],
        Flux.ConvTranspose(ones(Float32, windowsize, 1, 1))(sum(spec[40:100,:], dims=1)'[:,:,:])[:,1,1],
        Flux.ConvTranspose(ones(Float32, windowsize, 1, 1))(sum(spec[200:end,:], dims=1)'[:,:,:])[:,1,1],
        dims=2
    )

    initial_latent ./= Flux.ConvTranspose(ones(Float32, windowsize, 1, 1))(ones(Float32, fs-(windowsize-1), 1, 1))[:,1,1]

    initial_latent .-= Statistics.mean(initial_latent, dims=1)
    initial_latent ./= Statistics.std(initial_latent, dims=1)

    f = DSP.digitalfilter(DSP.Lowpass(40/(fs/2)), DSP.FIRWindow(DSP.hanning(10000)))
    initial_latent = DSP.filtfilt(f, initial_latent[:,:,1])

    initial_latent .-= Statistics.mean(initial_latent, dims=1)
    initial_latent ./= Statistics.std(initial_latent, dims=1)
    initial_latent ./= maximum(abs.(initial_latent), dims=1)

    initial_latent = initial_latent[:,:,:]
    # initial_latent .*= 0.5

    initial_latent = initial_latent |> dev
    # =#

    # initial_latent = 1f-10 * randn(Float32, 48000, 3, 1) |> dev

    @info "Setting up latent"
    num_latent = 3

    n_sub = 8

    l_latent = Int(ceil((48000+12000) * (1/2)^n_sub))

    latent = 1f-5 .* randn(Float32, l_latent, num_latent, 1) |> dev

    @info "Setting up model..."
    latent_coeff = [25f0, 25f0]; latent_coeff ./= (5f-1 * sum(latent_coeff))

    m_latent = Flux.Chain([
        Flux.Chain(
            Flux.ConvTranspose(Float32.(repeat([1, 1], 1, 1, num_latent)), stride=2, groups=num_latent),
            Flux.Conv(Float32.(repeat(DSP.digitalfilter(DSP.Lowpass(0.25), DSP.FIRWindow(DSP.hamming(15))), 1, 1, num_latent)), groups=num_latent))
        for n in 1:n_sub]) |> dev

    latent_offset = div(size(m_latent(latent), 1) - 48000, 2)

    channels = 16
    m_amp = Flux.Chain(
        Flux.Conv((1, ), (1+num_latent)=>channels),
        [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
            Flux.Conv((1,), channels=>1)
            ) |> dev

    offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

    @info "Setting up optimizer..."
    # opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
    opt = Flux.setup(Flux.Adam(5f-4), [latent, m_amp])

    losses = []
    losses_mse = []
    losses_diff = []
    losses_var = []
    val_losses = []

    val_region = 24000:24100

    train_mask = ones(Float32, 48000, 1, 1) |> dev
    train_mask[val_region] .= 0

    # val_mask = 1f0 .- train_mask

    diff_weights = Float32.(0.1.^(0:(num_latent-1))') |> dev

    f2 = DSP.digitalfilter(DSP.Lowpass(100/48000), DSP.FIRWindow(DSP.hanning(10000)))
    
    function random_direction(D)
        ret = randn(Float32, D)
        ret ./= LinearAlgebra.norm(ret)
    end
    
    function sample_characteristic_function(samples, evaluation_points)
        (1/length(samples) * sum(exp.(-im .* evaluation_points' .* samples), dims=1))
    end
end

@info "Training..."
# for n in 1:150000
n = 0
while true
    global losses
    global val_losses
    global n
    global loss

    current_latent = m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:] .+ initial_latent |> Flux.cpu
    current_latent_filtered = DSP.filtfilt(f2, current_latent) |> dev

    grid = 2 * (rand(Float32, 1, num_latent, 30) .- 0.5) |> dev

    ds = [random_direction(D) for n in 1:10] |> dev
    evaluation_range = 50
    evaluation_points = evaluation_range .* (rand(10, 1) .- 0.5) |> dev
    target_samples = 2 * rand(Float32, 50, D) .- 1 |> dev
    # target_samples = randn(Float32, 50, D) |> dev
    targets = [sample_characteristic_function(target_samples * d, evaluation_points) for d in ds]

    lp = Random.rand(1:48000, 100)
    loss, grads = Flux.withgradient([latent, m_amp]) do (latent, m_amp, )
        latent_smooth = m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:] .+ initial_latent

        y = m_amp(cat(latent_smooth, x_dev, dims=2))

        l_mse = (length(train_mask) / sum(train_mask)) * Flux.mse(
            train_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            train_mask[offset:end,:,:] .* y)

        l_filt = 0f0 # * Statistics.mean((current_latent_filtered .- latent_smooth).^2)
        l_mean = 0f0 # * Statistics.mean(Statistics.mean(latent_smooth, dims=1).^2)
        l_diff = 0f0 # .* Float32(fs) * Statistics.mean(diff_weights .* diff(latent_smooth, dims=1).^2)
        l_var = 0f0 # * Statistics.mean((1f-0 .- Statistics.var(latent_smooth, dims=1)).^2)
        l_cov = 0f0 # * Statistics.mean((I .- cov(latent_smooth)).^2)

        #=
        l_val = (length(val_mask) / sum(val_mask)) * Flux.mse(
            val_mask[offset:end,:,:] .* y_dev[offset:end, :, :],
            val_mask[offset:end,:,:] .* y)
        =#

        l_val = Flux.mse(y[val_region, :, :], y_dev[val_region .+ offset .- 1, :, :])

        ls = latent_smooth[lp,:,:]
        # l_grid = 1f-4 * (Statistics.mean(minimum(sum((ls .- grid).^2, dims=2), dims=1)) + Statistics.mean(minimum(sum((ls .- grid).^2, dims=2), dims=3)))

        l_grid = 1f-0 * Statistics.mean([Flux.mse(targets[n][:], sample_characteristic_function(ls[:,:] * ds[n], evaluation_points)[:]) for n in 1:length(ds)])

        (
            l_mse + l_mean + l_diff + l_var + l_filt + l_cov + 1f-2 * l_grid,
            l_mse, l_diff, l_var, l_cov, l_mean, l_filt, l_grid, l_val
        )
    end
    Flux.update!(opt, [latent, m_amp], grads[1])


    # latent .*= (1f0 - 1f-2)

    push!(losses, loss[1])
    push!(losses_mse, loss[2])
    push!(val_losses, loss[end])

    @info "n: $n, losses: $(loss)"

    # Flux.update!(opt, latent2, grads[1])

    if n % 100 == 0
        Plots.plot(
            Plots.plot(y, label="y"),
            Plots.plot(Flux.cpu(m_latent(latent)[latent_offset:(latent_offset+48000-1),:,1] .+ initial_latent[:,:,1]), label="combined latent"),
            Plots.plot(Flux.cpu(m_latent(latent)[latent_offset:(latent_offset+48000-1),:,1]), label="latent correction"),
            Plots.plot(Flux.cpu(initial_latent[:,:,1]), label="initial latent"),
            Plots.plot(controls, label="controls"),
            Plots.plot(log10.(losses), label="log10(training loss)"),
            Plots.plot(log10.(losses_mse), label="log10(mse)"),
            # Plots.plot(log10.(losses_diff)),
            # Plots.plot(log10.(losses_var)),
            Plots.plot(log10.(val_losses), label="log10(validation loss)"),
            layout=(8,1)
            ) |> display

        # @info "losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent), x_dev, dims=2)))), cov: $(1f-1 * sum((I .- cov(m_latent(latent))).^2))"
        # @info "losses: amp: $(Flux.mse(y_dev[offset:end, :, :], m_amp(cat(m_latent(latent)[latent_offset:(latent_offset+48000-1),:,:], x_dev, dims=2))))"
    end
    n += 1
end
