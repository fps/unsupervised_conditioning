
@info "Reading training data..."

# y, y_fs = WAV.wavread("data/test/Take1_Audio 1-1_AmpVTS_CabinetIV.wav");
# y, y_fs = WAV.wavread("data/synthetic/AmpVTS/nam_input_add_noise_output.wav");
# y, y_fs = WAV.wavread("signal_output.wav");
# y, y_fs = WAV.wavread("data/synthetic/AmpVTS/nam_input_spliced_noise_output.wav");
y = Float32.(output)

y = Float32.(y)

y_mean = Statistics.mean(y)
y_std = Statistics.std(y)

y .-= y_mean
y ./= y_std

# x, x_fs = WAV.wavread("data/test/Take1_Audio 1-1.wav");
# x, x_fs = WAV.wavread("data/nam_input_add_noise.wav");
# x, x_fs = WAV.wavread("signal.wav");
# x, x_fs = WAV.wavread("data/nam_input_spliced_noise.wav");

x = Float32.(input)

x = Float32.(x)

x_mean = Statistics.mean(x)
x_std = Statistics.std(x)

x .-= x_mean
x ./= x_std
# =#

@info "Creating chunks..."

chunk_hop = 6 * m_amp_offset
chunk_length = 6 * m_amp_offset
chunk_starts = 1:chunk_hop:(length(x)-(chunk_length-1))

chunks_x = cat([x[(start + m_amp_offset):(start + 3 * m_amp_offset - 1)] for start in chunk_starts]..., dims=3)
chunks_y = cat([y[(start + m_amp_offset):(start + 3 * m_amp_offset - 1)] for start in chunk_starts]..., dims=3)

@info "Calculating spectra..."

# chunks_x_fft = cat([spectrum(x[(start + 4 * m_amp_offset):(start + 6 * m_amp_offset - 1)], fft_size, fft_hop) for start in chunk_starts]..., dims=2)
chunks_y_fft = cat([spectrum(y[(start + 4 * m_amp_offset):(start + 6 * m_amp_offset - 1)], fft_size, fft_hop) for start in chunk_starts]..., dims=2)

# STAHP

# chunks_x_fft .-= Statistics.mean(chunks_x_fft, dims=2)
# chunks_x_fft ./= (0.0001 .+ Statistics.std(chunks_x_fft, dims=2))

chunks_y_fft .-= Statistics.mean(chunks_y_fft, dims=2)
chunks_y_fft ./= (0.0001 .+ Statistics.std(chunks_y_fft, dims=2))

# chunks_x_fft ./= [2 * n for n in 1:div(fft_size, 2)]
# chunks_y_fft ./= [2 * n for n in 1:div(fft_size, 2)]

# N = div(fft_size,2); starts = Int.(floor.([N/(2^(n+1)) for n in 1:(log2(N)-1)])); ends = Int.(floor.([N/2^n for n in 1:(log2(N)-1)]))

chunks_supervision = zeros(Float32, 4, length(chunk_starts))

for n in 1:length(supervised_control_points)
    chunks_supervision[1, (6*(n-1)) .+ (5:6)] .= 1
    chunks_supervision[2:4, (6*(n-1)) + 5] = supervised_control_points[n]
    chunks_supervision[2:4, (6*(n-1)) + 6] = supervised_control_points[n]
end

#=
for n in 1:length(chunk_starts)
    if n <= 2 * length(supervised_controls)
        chunks_supervision[1, n] = 1
        chunks_supervision[2:4, n] = supervised_controls[div(n+1, 2)]
    end
end
=#

# chunks_supervision = [ if n > 2*length(supervised_controls); nothing; else; supervised_controls[div(n+1, 2)]; end for n in 1:length(chunk_starts) ]
# chunks_supervision = reshape(chunks_supervision, 1, length(chunks_supervision))

chunks = (
    chunks_x[:,:,1:(end-1)],
    chunks_y[:,:,1:(end-1)],
    chunks_x[:,:,2:end],
    chunks_y[:,:,2:end],
    # chunks_x_fft[:,1:(end-1)],
    chunks_y_fft[:,1:(end-1)],
    # chunks_x_fft[:,2:end],
    chunks_y_fft[:,2:end],
    chunks_supervision[:,1:(end-1)],
    chunks_supervision[:,2:end]
)

chunks_dev = chunks |> dev

@info "Done."

