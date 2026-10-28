
ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "8GiB"

import Pkg

@info "Activating package..."
Pkg.activate(".")

@info "Importing packages..."
import CUDA, cuDNN, Flux, WAV, Plots, DSP, Statistics, Random, ProgressBars, UnicodePlots, FFTW, MultivariateStats, DelimitedFiles, LinearAlgebra

# Plots.unicodeplots()
Plots.default(dpi=300, linewidth=0.5, alpha=0.75, size=(1000, 1200))

dev = Flux.gpu
cpu = Flux.cpu


include("../AmpModeling.jl/src/AmpModeling.jl")
include("util.jl")
# include("chunks.jl")
include("time_encodings.jl")
include("synthetic_data.jl")

@info "Reading training data..."

# y, y_fs = WAV.wavread("data/test/Take1_Audio 1-1_AmpVTS_CabinetIV.wav");
y, y_fs = WAV.wavread("data/synthetic/AmpVTS/nam_input_add_noise_output.wav");

y = Float32.(y)

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)

y .-= y_mean
y ./= y_std

# x, x_fs = WAV.wavread("data/test/Take1_Audio 1-1.wav");
x, x_fs = WAV.wavread("data/nam_input_add_noise.wav");

x = Float32.(x)

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)

x .-= x_mean
x ./= x_std
# =#

chunk_hop = 1024
chunk_length = 8192
chunk_starts = 1:chunk_hop:(length(x)-(chunk_length-1))

chunks = (
    cat([x[start:(start+chunk_length-1)] for start in chunk_starts[1:(end-1)]]..., dims=3),
    cat([x[start:(start+chunk_length-1)] for start in chunk_starts[2:end]]..., dims=3),
    cat([y[start:(start+chunk_length-1)] for start in chunk_starts[1:(end-1)]]..., dims=3),
    cat([y[start:(start+chunk_length-1)] for start in chunk_starts[2:end]]..., dims=3)
)

chunks_dev = chunks |> dev

m_encoder = Flux.Chain(
    Flux.Conv((1, ), 2 => 16),
    [ Flux.Conv((3,), 16 => 16, Flux.leakyrelu, pad=Flux.SamePad(), stride=2) for n in 1:13]...,
    Flux.Conv((1, ), 16 => 3)
) |> dev

m_amp = Flux.Chain(
    Flux.Conv((1, ), 4 => 16),
    [ Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:11 ]...,
    Flux.Conv((1,), 16 => 1)
) |> dev

m_amp_offset = 48000 - size(m_amp(dev(randn(Float32, 48000, 4, 1))), 1)

@info "Setting up optimizer..."

opt = Flux.setup(Flux.Adam(), [m_encoder, m_amp])

epoch = 1

losses = []
losses_mse = []

while true
    global epoch
    global losses
    global encoded

    data_loader = Flux.MLUtils.DataLoader(chunks_dev, batchsize=8, shuffle=true, parallel=false)

    epoch_losses = []
    epoch_losses_mse = []
    
    for (x1, x2, y1, y2) in ProgressBars.ProgressBar(data_loader)
        
        loss, grad = Flux.withgradient([m_encoder, m_amp]) do (m_encoder, m_amp)
            encoded1 = m_encoder(cat(x1, y1, dims=2))
            encoded2 = m_encoder(cat(x2, y2, dims=2))
            
            l_similarity = 1f-0 * Flux.mse(encoded1, encoded2)
            l_mean = Statistics.mean([Statistics.mean(encoded1.^2), Statistics.mean(encoded2.^2)])
            
            processed1 = m_amp(cat(x1, repeat(encoded1, chunk_length, 1, 1), dims=2))
            processed2 = m_amp(cat(x2, repeat(encoded2, chunk_length, 1, 1), dims=2))

            l_mse1 = Flux.mse(y1[(1+m_amp_offset):end, :, :], processed1)
            l_mse2 = Flux.mse(y2[(1+m_amp_offset):end, :, :], processed2)
            
            (l_similarity + l_mean + l_mse1 + l_mse2, l_similarity, l_mean, l_mse1, l_mse2)
        end
        
        Flux.update!(opt, [m_encoder, m_amp], grad[1])
        
        push!(epoch_losses, loss[1])
        push!(epoch_losses_mse, Statistics.mean(loss[4:5]))
    end
    
    push!(losses, Statistics.mean(epoch_losses))
    push!(losses_mse, Statistics.mean(epoch_losses_mse))    
              
    encoded = cat([m_encoder(cat(chunks_dev[1][:,:,n:n], chunks_dev[3][:,:,n:n], dims=2)[:,:,:])[1,:,1] for n in 1:size(chunks_dev[1], 3)]..., dims=2)

    for n in 1:3; UnicodePlots.lineplot(encoded[n,:] |> cpu, width=:auto) |> display; end
    
    UnicodePlots.lineplot(log10.(losses), width=:auto) |> display
    UnicodePlots.lineplot(log10.(losses_mse), width=:auto) |> display
    
    epoch += 1

    if epoch >= 10000
        break
    end
end
