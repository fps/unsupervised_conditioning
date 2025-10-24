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

for n in 1:length(controls)
    controls[n] = collect(controls[n])
    controls[n] .-= Statistics.mean(controls[n])
end

# Random.seed!(2)

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

num_latent = 3

channels = 16
# activation = Flux.rrelu
# activation = Flux.leakyrelu
activation = Flux.tanh

m_decoder = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, activation, dilation=2^k) for k in 0:4]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 48000 - size(m_decoder(randn(Float32, 48000, 4, 1) |> dev), 1)

input = Float32.(cat(x, controls..., dims=2)) |> dev

lr = 1e-3
schedule = ParameterSchedulers.Stateful(ParameterSchedulers.CosAnneal(lr, 1e-4, 5000))

println("Setting up optimizer...")
# opt = Flux.setup(Flux.Adam(2f-4), [m_encoder, m_latent_down, m_latent_up, m_decoder])
opt = Flux.setup(Flux.Adam(lr), m_decoder)
# opt = Flux.setup(Flux.AdamW(eta=2f-4, lambda=0.00001), [m_encoder, m_decoder])

loss_min = 1f10
losses = []

Plots.default(dpi=300, linewidth=0.5, alpha=0.75)

println("Training...")
for n in 1:50000
    global loss_min
    global losses
    # loss, grads = Flux.withgradient([m_encoder, m_latent_down, m_latent_up, m_decoder]) do params
    loss, grads = Flux.withgradient(m_decoder) do m_decoder
        Flux.mse(y_dev[(1+offset):end, :, :], m_decoder(input[:,:,:]))
    end

    if loss < loss_min
        global m_min = deepcopy(m_decoder)
        loss_min = loss
    end

    global lr = ParameterSchedulers.next!(schedule)
    # Flux.adjust!(opt, lr)
    Flux.adjust!(opt, lr + 1f-4 * abs(randn()))
    println("n: $n, lr: $lr, loss: $loss")
    # Flux.update!(opt, [m_encoder, m_latent_down, m_latent_up, m_decoder], grads[1])
    Flux.update!(opt, m_decoder, grads[1])

    push!(losses, loss)

    if n % 100 == 0
        Plots.plot(
            Plots.plot(y[:,1,1] |> Flux.cpu, label="Target"),
            Plots.plot((input[:,:,:] |> m_decoder |> Flux.cpu)[:,1,1], label="Prediction"),
            Plots.plot(log10.(losses), label="Loss (log10)"),
            layout=(3,1),
            linewidth=0.2
        ) |> display
    end
    # CUDA.reclaim()
end
