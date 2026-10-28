ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "6GiB"

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
include("spectrum.jl")
include("sigreg.jl")

@info "Reading training data..."

# y, y_fs = WAV.wavread("data/test/Take1_Audio 1-1_AmpVTS_CabinetIV.wav");
# y, y_fs = WAV.wavread("data/synthetic/AmpVTS/nam_input_add_noise_output.wav");
# y, y_fs = WAV.wavread("signal_output.wav");
y, y_fs = WAV.wavread("data/synthetic/AmpVTS/nam_input_spliced_noise_output.wav");

y = Float32.(y)

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)

y .-= y_mean
y ./= y_std

# x, x_fs = WAV.wavread("data/test/Take1_Audio 1-1.wav");
# x, x_fs = WAV.wavread("data/nam_input_add_noise.wav");
# x, x_fs = WAV.wavread("signal.wav");
x, x_fs = WAV.wavread("data/nam_input_spliced_noise.wav");

x = Float32.(x)

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)

x .-= x_mean
x ./= x_std
# =#

@info "Setting up model..."

fft_size = 1024
fft_hop = div(fft_size, 2)

m_encoder = Flux.Chain(
    Flux.Dense(fft_size => 64),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    # Flux.BatchNorm(64),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    # Flux.BatchNorm(64),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    # Flux.BatchNorm(64),
    Flux.Dense(64 => 32, Flux.leakyrelu),
    Flux.Dense(32 => 16, Flux.leakyrelu),
    Flux.Dense(16 => 16, Flux.leakyrelu),
    # Flux.Dense(16 => 8, Flux.tanh),
    Flux.Dense(16 => 3),
) |> dev

m_amp = Flux.Chain(
    Flux.Conv((1, ), 4 => 16),
    Flux.SkipConnection(Flux.Chain([Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:10 ]...), (mx, x) -> (mx .+ x[(size(x, 1) - size(mx, 1) + 1):end, :, :])),
    [Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:10 ]...,
    Flux.Conv((1,), 16 => 1)
) |> dev

m_amp_offset = 48000 - size(m_amp(dev(randn(Float32, 48000, 4, 1))), 1)

@info "m_amp_offset: $m_amp_offset"

@info "Creating chunks..."

chunk_hop = 6 * m_amp_offset
chunk_length = 6 * m_amp_offset
chunk_starts = 1:chunk_hop:(length(x)-(chunk_length-1))

chunks_x = cat([x[(start + m_amp_offset):(start + 3 * m_amp_offset - 1)] for start in chunk_starts]..., dims=3)
chunks_y = cat([y[(start + m_amp_offset):(start + 3 * m_amp_offset - 1)] for start in chunk_starts]..., dims=3)

@info "Calculating spectra..."

chunks_x_fft = cat([spectrum(x[(start + 4 * m_amp_offset):(start + 6 * m_amp_offset - 1)], fft_size, fft_hop) for start in chunk_starts]..., dims=2)
chunks_y_fft = cat([spectrum(y[(start + 4 * m_amp_offset):(start + 6 * m_amp_offset - 1)], fft_size, fft_hop) for start in chunk_starts]..., dims=2)

# STAHP

chunks_x_fft .-= Statistics.mean(chunks_x_fft, dims=2)
chunks_x_fft ./= (0.0001 .+ Statistics.std(chunks_x_fft, dims=2))

chunks_y_fft .-= Statistics.mean(chunks_y_fft, dims=2)
chunks_y_fft ./= (0.0001 .+ Statistics.std(chunks_y_fft, dims=2))

# chunks_x_fft ./= [2 * n for n in 1:div(fft_size, 2)]
# chunks_y_fft ./= [2 * n for n in 1:div(fft_size, 2)]

# N = div(fft_size,2); starts = Int.(floor.([N/(2^(n+1)) for n in 1:(log2(N)-1)])); ends = Int.(floor.([N/2^n for n in 1:(log2(N)-1)]))

chunks = (
    chunks_x[:,:,1:(end-1)],
    chunks_y[:,:,1:(end-1)],
    chunks_x[:,:,2:end],
    chunks_y[:,:,2:end],
    chunks_x_fft[:,1:(end-1)],
    chunks_y_fft[:,1:(end-1)],
    chunks_x_fft[:,2:end],
    chunks_y_fft[:,2:end]
)

chunks_dev = chunks |> dev

@info "Setting up optimizer..."

opt = Flux.setup(Flux.Adam(2f-4), [m_encoder, m_amp])

# Flux.adjust!(opt[1], 2f-5)

@info "Training..."

epoch = 1

losses = []

# similarity_weights = dev([1 3 9]')
similarity_weights = dev([1 1 1]')


# STAHP

sigreg_test_points = dev(Float32.(collect(range(-5, 5, 17))))
sigreg_N_slices = 16

while true
    global epoch
    global losses
    global losses_mse

    data_loader = Flux.MLUtils.DataLoader(chunks_dev, batchsize=32, shuffle=true, partial=false)

    epoch_losses = []
    
    for (x1, y1, x2, y2, x1_spec, y1_spec, x2_spec, y2_spec) in ProgressBars.ProgressBar(data_loader)
        directions = dev(randn(Float32, sigreg_N_slices, 3))
        
        for n in 1:sigreg_N_slices
            directions[n,:] ./ LinearAlgebra.norm(directions[n,:])
        end
        
        loss, grad = Flux.withgradient([m_encoder, m_amp]) do (m_encoder, m_amp)
            encoded1 = m_encoder(cat(x1_spec, y1_spec, dims=1))
            encoded2 = m_encoder(cat(x2_spec, y2_spec, dims=1))
            
            l_similarity = 1f1 * Flux.mse(similarity_weights .* encoded1, similarity_weights .* encoded2)
            # l_encoded_mean = 1f-0 * Statistics.mean(cat(Statistics.mean(encoded1, dims=2).^2, Statistics.mean(encoded2, dims=2).^2, dims=1))
            # l_var = 0f-1 * Statistics.mean((1 .- Statistics.std(cat(encoded1, encoded2, dims=2), dims=2)).^2)

            l_sigreg = 1f-1 * SIGReg.sigreg(cat(encoded1, encoded2, dims=2), sigreg_test_points, directions)

            #=
            processed1 = m_amp(cat(x1, permutedims(repeat(encoded1, 1, 1, size(x1, 1)), (3, 1, 2)), dims=2)[:,:,:])
            processed2 = m_amp(cat(x2, permutedims(repeat(encoded2, 1, 1, size(x2, 1)), (3, 1, 2)), dims=2)[:,:,:])

            l_mse1 = Flux.mse(y1[(1+m_amp_offset):end, :, :], processed1)
            l_mse2 = Flux.mse(y2[(1+m_amp_offset):end, :, :], processed2)

            if isnan(l_similarity); error("nan - similarity"); end
            if isnan(l_encoded_mean); error("nan - mean"); end
            # if isnan(l_var); error("nan - var"); end
            if isnan(l_mse1); error("nan - mse1"); end
            if isnan(l_mse2); error("nan - mse2"); end
            =#

            # l_mse1 = 0
            # l_mse2 = 0
            (
                l_similarity #= + l_var =# + l_sigreg  #=  l_encoded_mean + (l_mse1 + l_mse2)/2 =#,
                l_similarity,
                l_sigreg,
                # l_encoded_mean,
                # (l_mse1 + l_mse2) / 2
            )
        end
        
        Flux.update!(opt, [m_encoder, m_amp], grad[1])

        push!(epoch_losses, loss)
    end

    N_losses = length(epoch_losses[1])
    
    mean_losses = zeros(N_losses)
    for l in 1:length(epoch_losses[1])
        mean_losses[l] = Statistics.mean([epoch_losses[k][l] for k in 1:length(epoch_losses)])
    end
    
    push!(losses, mean_losses)
    
    if (epoch - 1) % 10 == 0
        for n in 1:3
            UnicodePlots.lineplot(cpu(m_encoder(cat(chunks_dev[5], chunks_dev[6], dims=1)))[n,:], width=:auto) |> display
        end
    
        # UnicodePlots.lineplot(log10.(losses), width=:auto) |> display
        for n in 1:length(losses[1])
            UnicodePlots.lineplot(log10.([losses[k][n] for k in 1:length(losses)]), width=:auto) |> display
        end
    end
    
    epoch += 1

    if epoch >= 10000
        break
    end
end
