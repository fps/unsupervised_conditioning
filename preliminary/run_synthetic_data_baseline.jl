ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"] = "6GiB"

import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random, ParameterSchedulers

# import AmpModeling


dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")
include("half_band_filters.jl")

println("Generating synthetic data...")
x, y, controls = synthetic_data_3controls1()

x_mean = Statistics.mean(x, dims=1)
x_std = Statistics.std(x, dims=1)

y_mean = Statistics.mean(y, dims=1)
y_std = Statistics.std(y, dims=1)

x .-= x_mean
x ./= x_std

y .-= y_mean
y ./= y_std

#=
for n in 1:length(controls)
    controls[n] = collect(controls[n])
    controls[n] .-= Statistics.mean(controls[n])
end
=#

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

x2, y2, controls2 = synthetic_data_3controls2()

x2 .-= x_mean
x2 ./= x_std

y2 .-= y_mean
y2 ./= y_std


#=
for n in 1:length(controls2)
    controls2[n] = collect(controls2[n])
    controls2[n] .-= Statistics.mean(controls2[n])
end
=#

x2_dev = x2 |> dev
y2_dev = repeat(y2, 1, 1, 1) |> dev

# Random.seed!(2)

Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

num_latent = 3

channels = 32
# activation = Flux.rrelu
# activation = Flux.leakyrelu
activation = Flux.tanh

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, activation, dilation=2^k) for k in 0:5]...,
    # [AmpModeling.AdditiveSkipConnection(Flux.Conv((3,), channels=>channels, activation, dilation=2^k)) for k in 0:5]...,
    # [AmpModeling.AveragingSkipConnection(Flux.Conv((3,), channels=>channels, activation, dilation=2^k)) for k in 0:5]...,

    Flux.Conv((1,), channels=>1)
) |> dev

offset = 48000 - size(m_decoder(randn(Float32, 48000, 4, 1) |> dev), 1)

input = Float32.(cat(x, controls..., dims=2)) |> dev
input2 = Float32.(cat(x2, controls2..., dims=2)) |> dev

lr = 1e-3
schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(lr, 1e-4, 25000))

println("Setting up optimizer...")
# opt = Flux.setup(Flux.Adam(2f-4), [m_encoder, m_latent_down, m_latent_up, m_decoder])
opt = Flux.setup(Flux.Adam(lr), m_decoder)
# opt = Flux.setup(Flux.AdamW(eta=2f-4, lambda=0.00001), [m_encoder, m_decoder])

loss_min = 1f10
losses = []
losses_val = []

Plots.default(dpi=300, linewidth=0.5, alpha=0.75)

println("Training...")
for n in 1:50000
    global loss_min
    global losses
    # loss, grads = Flux.withgradient([m_encoder, m_latent_down, m_latent_up, m_decoder]) do params
    loss, grads = Flux.withgradient(m_decoder) do m_decoder
        Flux.mse(y_dev[(1+offset):end, :, :], m_decoder(input[:,:,:]))
    end

    loss_val = Flux.mse(y2_dev[(1+offset):end, :, :], m_decoder(input2[:,:,:]))

    if loss < loss_min
        global m_min = deepcopy(m_decoder)
        loss_min = loss
    end

    global lr = ParameterSchedulers.next!(schedule)
    Flux.adjust!(opt, lr)
    # Flux.adjust!(opt, lr + 1f-4 * abs(randn()))
    println("n: $n, lr: $lr, loss: $loss")
    # Flux.update!(opt, [m_encoder, m_latent_down, m_latent_up, m_decoder], grads[1])
    Flux.update!(opt, m_decoder, grads[1])

    push!(losses, loss)
    push!(losses_val, loss_val)

    if n % 100 == 0
        #=
        # Plots.plot(
            # Plots.plot(y[:,1,1] |> Flux.cpu, label="Target"),
            # Plots.plot((input[:,:,:] |> m_decoder |> Flux.cpu)[:,1,1], label="Prediction (Training)"),
            # Plots.plot((input2[:,:,:] |> m_decoder |> Flux.cpu)[:,1,1], # label="Prediction (Validation)"),
            Plots.plot(log10.(losses), title="Training loss (log10)"),
            Plots.plot!(log10.(losses_val), title="Validation loss (log10)"),
            layout=(2,1),
            linewidth=0.2
        # ) |> display
        =#
        Plots.plot(20. * log10.(losses), label="Training loss (dB)")
        Plots.plot!(20 .* log10.(losses_val), label="Validation loss (dB)") |> display
    end
    # CUDA.reclaim()
end
