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
x, y, controls = synthetic_data_3controls1()
fs = 48000

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(x), Plots.plot(y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Generating basis functions")

n_basis = 100
basis = Float32.(hcat((0:(1/48000):(1-(1/48000))) .- 0.5, [sin.(phase .+ 2*pi*k .* (0:(1/fs):(1-(1/fs)))) for k in 1:n_basis for phase in [0, pi/2]]...)) |> dev

println("Setting up latent")
num_latent = 6

# basis_weights = Float32.(1 ./ repeat(n_basis:-1:1, inner=2))  |> dev
# basis_weight = Float32.(repeat(1:n_basis, inner=2)) |> dev
# basis_weights = Float32.(sqrt.(repeat(1:n_basis, inner=2)) ./ n_basis) |> dev

# basis_weights ./= Statistics.mean(basis_weights)
cutoff = 20
basis_weights = cat([1], ones(cutoff*2), repeat(1:(n_basis-cutoff), inner=2), dims=1) |> dev

latent = 1f-5 .* randn(Float32, n_basis * 2 + 1, num_latent) ./ (2*n_basis + 1) |> dev

# latent ./= (basis_weights .+ 1)

println("Setting up model...")
# m_latent = Flux.Conv(repeat(coeff, 1, 1, num_latent), groups=num_latent) |> dev
m_latent(latent) = basis * latent

channels = 16
m_amp = Flux.Chain(
    Flux.Conv((1, ), (1+num_latent)=>channels),
    [Flux.Conv((3,), channels=>channels, Flux.tanh, dilation=2^k) for k in 0:8]...,
    Flux.Conv((1,), channels=>1)
) |> dev

offset = 1 + 48000 - size(m_amp(randn(Float32, 48000, 1+num_latent, 1)|>dev), 1)

println("Setting up optimizer...")
# opt = Flux.setup(Flux.AdamW(lambda=0.2), [latent, m_amp])
opt = Flux.setup(Flux.Adam(1f-4), [latent, m_amp])

I = sqrt(1f-2) .* eye(num_latent)

losses = []
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
        latent_smooth = m_latent(params[1])[:,:,:]
        l_mse = Flux.mse(y_dev[offset:end, :, :], params[2](cat(latent_smooth, x_dev, dims=2)))
        l_weights = 1f0 * Statistics.mean((params[1].^2 .* basis_weights))
        (l_mse + l_weights, l_mse, l_weights)
    end
    Flux.update!(opt, [latent, m_amp], grads[1])

    push!(losses, loss[1])
    push!(losses_mse, loss[2])

    if n % 100 == 0
        @info "n: $n, losses: $loss"
        Plots.plot(
            Plots.plot(y),
            Plots.plot(Flux.cpu(m_latent(latent))[:,:,1]),
            Plots.plot(controls),
            Plots.plot(log10.(losses)),
            Plots.plot(log10.(losses_mse)),
            layout=(5,1),
            linewidth=0.1,
            legend=:none
        ) |> display
    end
    n += 1
end
