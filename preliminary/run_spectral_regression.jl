import Pkg
Pkg.activate(".")

import Flux, CUDA, cuDNN, DSP, UnicodePlots, Statistics

include("time_encodings.jl")
include("synthetic_data.jl")

x, y, controls = synthetic_data_3controls1()

n_bins = 33
n_window = 2 * (n_bins - 1)

specx = Float32.(log10.(DSP.spectrogram(x, n_window).power'))[:,:,:]
specy = Float32.(log10.(DSP.spectrogram(y, n_window).power'))[:,:,:]

n_latent = 3

latent = 1f-5 .* randn(Float32, size(specx, 1), n_latent, 1)

specm = Flux.Chain(
    Flux.Conv((3,), (n_bins + n_latent) => n_bins),
    [ Flux.Conv((3,), n_bins => n_bins, Flux.leakyrelu) for n in 1:3]...,
    Flux.Conv((3,), n_bins => n_bins))

opt = Flux.setup(Flux.Adam(), [ latent, specm ])

epoch = 0
while true
    global epoch
    l, g = Flux.withgradient([latent, specm]) do p
        l = Flux.mse(p[2](cat(specx, p[1], dims=2)), specy[1:1489, :, :])
        l += 1f3 * Statistics.mean(diff(p[1], dims=1).^2)
    end

    @info "epoch: $(epoch), loss: $(l)"
    Flux.update!(opt, [latent, specm], g[1])

    if epoch % 100 == 0
        for n in 1:n_latent
            UnicodePlots.lineplot(latent[:,n,1], width=:auto) |> display
        end
    end

    epoch += 1
end
