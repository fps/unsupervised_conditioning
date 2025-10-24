import Pkg

println("Activating package...")
Pkg.activate(".")

println("Importing packages...")
import CUDA, Flux, WAV, Plots, DSP, Statistics, Random

dev = Flux.gpu

include("synthetic_data.jl")
include("util.jl")
# include("basis.jl")

println("Generating synthetic data...")
x, y, fs, sigmas, controls, t = synthetic_data_3controls()

x_dev = x |> dev
y_dev = repeat(y, 1, 1, 1) |> dev

Plots.plot(Plots.plot(t, x), Plots.plot(t,y), Plots.plot(controls), layout=(3,1), linewidth=0.1, fontsize=1) |> display

println("Designing lowpass filter...")
ord = DSP.remezord(100/fs, 150/fs, 0.1, 0.001)
coeff = Float32.(DSP.digitalfilter(DSP.Lowpass(10), DSP.FIRWindow(DSP.Windows.hann(ord)), fs=fs))

println("Setting up latents")
D_latent = 3
N_latent = 100

latent = (rand(Float32, N_latent, D_latent) .- 0.5) |> dev
latent_t = Float32.(Random.rand(1:size(x, 1), N_latent)) |> dev

function gaussian(center, sigma, t)
    exp.(-(t.-center).^2 ./ (2*sigma.^2))
end

function nwe(sigma, x, y, x_eval)
    nom = 0
    denom = 0

    kv = gaussian(x, sigma, x_eval)
    sum(kv .* y, dims=1) ./ sum(kv, dims=1)
end
