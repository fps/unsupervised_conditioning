import Pkg
Pkg.activate(".")

import Flux, CUDA, cuDNN, DSP, Statistics, ProgressLogging, Images, BSON, Dates, Random, WAV

c = Flux.cpu
g = Flux.gpu

@info "# Utility functions"

function plot(data, ; width=680, height=300, opacity=0.5)
    img = 0.95 .* ones(Float32, height, width)
    img[1,:] .= 1f0
    img[end,:] .= 1f0
    img[:,1] .= 1f0
    img[:,end] .= 1f0
    data_max = maximum(abs.(data))
    # data_center = (data_min + data_max) / 2

    view_half_height = height / 2
    view_center = height / 2

    N_data = length(data)
    
    for n in 2:N_data
        d1 = data[n-1]
        d2 = data[n]
        
        col1 = 1+round((width-1)*(n-1.5)/N_data)
        col2 = 1+round((width-1)*(n-0.5)/N_data)
        
        row1 = 1+round(view_center - (view_half_height-1) * (d1 / data_max))
        row2 = 1+round(view_center - (view_half_height-1) * (d2 / data_max))

        N_steps = max(max(abs(row2-row1), (col2-col1)),1)
        #@info N_steps
        for step in 0:(1/N_steps):1
            row = Int(round(row1+(row2-row1)*step))
            col = Int(round(col1+(col2-col1)*step))
            # @info "row: $(row), col: $(col)"
            img[row, col] *= (1-opacity)
        end    
    end
    Images.Gray.(img)
end

function scatter(data; opacity = 0.5, width=680, height=300, markersize=3)
    img = 0.95*ones(Float32, height, width)
    data_max = maximum(abs.(data))
    # data_center = (data_min + data_max) / 2

    view_half_height = height / 2
    view_center = height / 2

    N_data = length(data)
    
    for n in 1:N_data
        d = data[n]
        col_center = 1+Int(round((width-1)*(n-0.5)/N_data))
        row_center = 1+Int(round(view_center - (view_half_height-1) * (d / data_max)))

        cols = max(col_center-(markersize-1),1):min(col_center+(markersize-1), width)
        rows = max(row_center-(markersize-1),1):min(row_center+(markersize-1), height)
        
        img[rows, col_center] .-= opacity
        img[row_center, cols] .-= opacity
        img[row_center, col_center] += opacity
    end
    Images.Gray.(img)
end

function filtord(f_pass, f_stop, ripple, attenuation, fs)
    n = DSP.remezord(f_pass/fs, f_stop/fs, ripple, attenuation)
    if iseven(n)
        n += 1
    end
    n
end;

function upsampled_size(k, n_bp, n_lo, s)
    s -= (n_bp - 1)
    for n in 1:k; 
        s *= 2; s-= (n_lo - 1);
    end; 
    s
end;

function latent_size(k, n_bp, n_lo, s)
    for n in 1:k
        s += (n_lo - 1)
        s = ceil(s/2)
    end
    Int(s + (n_bp - 1))
end;

function filtering_model(coeff)
    Flux.Conv(Float32.(repeat(coeff, 1, 1, 1)))
end;

function upsampling_model(k, coeff)
    Flux.Chain([
        Flux.Chain(
            Flux.ConvTranspose([
                1.0f0; 1.0f0;;;], stride=2), 
            filtering_model(coeff)
            # Flux.Conv(Float32.(repeat(ones(n_lo)./n_lo, 1, 1, 1)))
        ) for n in 1:k
    ]...)
end;


function maximum_downsampling(fs, f)
    (fs / 2) / (2 * f)
end;


function halfings(d)
    Int(floor(log2(d)))
end;

@info "# Upsampling lowpass filter design"

begin
    lo_pass = 0.25
    lo_stop = 0.5
    n_lo = filtord(lo_pass, lo_stop, 0.1, 0.00001, 1)
end;

@info "n_lo: $(n_lo)"


lo_coeff = DSP.Filters.digitalfilter(
    DSP.Filters.Lowpass(
        lo_pass
    ), 
    DSP.Filters.FIRWindow(
        DSP.Windows.hann(n_lo)
    ),
    fs=1
);

plot(lo_coeff)

@info "# Composite latent model"

function latent_model(coeff_bp, k, coeff_lo)
    Flux.Chain(
        filtering_model(coeff_bp),
        upsampling_model(k, coeff_lo)
    )
end;

@info "# Training a model"
@info "## Real data (Fender Blues Junior)"
@info "#### Reading the training data..."

import WAV

fs_fender = 48000

T_start = Int(8*fs_fender)
T_end = Int(833*fs_fender)

begin
    output1, fs_output = WAV.wavread("../../data/fender_blues_junior/Take6_Audio 1-1.wav");  
    output = repeat(Float32.(output1[T_start:T_end]), 1, 1, 1) |> g
    (size(output), fs_output, length(output)/fs_output)
end

begin
    input1, fs_input = WAV.wavread("../../data/fender_blues_junior/varying_noise_20_min_seed_0.wav"); 
    input = repeat(Float32.(input1[T_start:T_end]), 1, 1, 1) |> g
    (size(input), fs_input, length(input)/fs_input)
end

begin
    @assert(fs_input == fs_output)
    fs_fender = fs_input
end

@info "#### Control data..."

control_periods = [
    [8*length(output) / fs_fender  (1/2)*length(output) / fs_fender],
    [(1/4)*length(output) / fs_fender  (1/100)*length(output) / fs_fender],
    [10 0.5],
]

control_frequencies = [1.0 ./ x for x in control_periods]

maximum_downsamplings = [
    [maximum_downsampling(fs_fender, max(1/10, maximum(c))) for c in control_frequencies]...
]

ks = halfings.(maximum_downsamplings)

downsamplings = [ 1 / k^2 for k in ks ]

@info "#### Filter design..."

bandpass_filter_ords = [
    filtord(
        maximum(control_frequencies[n]),
        2*maximum(control_frequencies[n]), 
        0.01, 
        0.01,
        fs_fender/(2^ks[n])
    )

    for n in 1:length(ks)
]

bandpass_filter_coeffs = [
    DSP.Filters.digitalfilter(
        DSP.Filters.Bandpass(control_frequencies[n]...), 
        DSP.Filters.FIRWindow(DSP.Windows.hann(bandpass_filter_ords[n])),
        fs = fs_fender / downsamplings[n]
    )
    for n in 1:length(ks)
]

lowpass_filter_ord = filtord(
    0.25,
    0.5,
    0.01,
    0.001,
    1
)

lowpass_filter_coeffs = DSP.Filters.digitalfilter(
    DSP.Filters.Lowpass(0.25),
    DSP.Filters.FIRWindow(DSP.Windows.hann(lowpass_filter_ord))
)


@info "# Latent models..."

latent_sizes = [
    latent_size(ks[n], bandpass_filter_ords[n], lowpass_filter_ord, length(output))

    for n in 1:length(ks)
]

latent_models = [
    latent_model(bandpass_filter_coeffs[n], ks[n], lowpass_filter_coeffs)
    
    for n in 1:length(ks)
] |> g;

# #=
@info "# Creating model and latent data"

latents = [
    1.0f-9 * randn(Float32, latent_sizes[n], 1, 1)

    for n in 1:length(ks)
] |> g;



begin
    amp_model_width=16
    
    amp_model = Flux.Chain(
        Flux.Conv((1,), (1+length(ks))=>amp_model_width),
        [Flux.Conv((3,), amp_model_width=>amp_model_width, Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
        [Flux.Conv((3,), amp_model_width=>amp_model_width, Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
        Flux.Conv((1,), amp_model_width=>1)
    ) |> g
end;
# =#

#=
@info "# Loading previously saved model and latent data"

saved = BSON.load("../upsampling_model2.bson")

latents = saved[:latents_cpu] |> g

amp_model = saved[:amp_model_cpu] |> g
=#


offset = 48000 - size(amp_model(randn(Float32, 48000, 4, 1) |> g), 1)


@info "#### Data splitting...."

N_segments = size(latents[1], 1) - (bandpass_filter_ords[1] + lowpass_filter_ord - 1)

@info "N_segments: $(N_segments)"


function data_latent(latents, n) 
    [
        latents[1][(1+n):(n + bandpass_filter_ords[1] + lowpass_filter_ord - 1), :, :],
        
        latents[2][(1+n*2^(ks[1]-ks[2])):((n+1)*2^(ks[1]-ks[2]) + bandpass_filter_ords[2] + lowpass_filter_ord - 2), :, :],
        
        latents[3][(1+n*2^(ks[1]-ks[3])):((n+1)*2^(ks[1]-ks[3]) + + bandpass_filter_ords[3] + lowpass_filter_ord - 2), :, :],
    ]
end

@info "# Training..."

opt = Flux.setup(Flux.Optimisers.AdamW(0.001), [latents, amp_model])

function train()
    perm = Random.randperm(N_segments)
    ProgressLogging.@progress for p in 1:N_segments
    # ProgressLogging.@progress for p in 1:1
    # for p in 1:N_segments
        n = perm[p] - 1
        loss, grad = Flux.withgradient([latents, amp_model]) do (latents, amp_model)
            local ls = data_latent(latents, n)
            
            local l = cat([latent_models[m](
                ls[m]) for m in 1:length(ks)]..., dims=2)
            
            local x = cat(
                input[(1+n*2^ks[1]):((n+1)*2^ks[1]), :, :], 
                l[1:(2^ks[1]), :, :], 
                dims=2
            )

            local y = output[(offset+1+n*2^ks[1]):((n+1)*2^ks[1]), :, :]
        
            local the_loss = Statistics.mean((amp_model(x) .- y).^2)

            the_loss += 1f-2 * Statistics.mean(l.^2)
            
            if n == 0
                # constrain very first latent to -1
                # the_loss += sum((-1 .- l[1, :, :]).^2)
            end
            
            if n == (N_segments - 1)
               # constrain very lat latent to 1
               # the_loss += sum((1 .- l[end,:,:]).^2)
            end
            
            the_loss
        end
        Flux.update!(opt, [latents, amp_model], grad[1])
        @info "loss: $(loss), step: $(p), segment: $(n)"
    end
    Images.save("latents.png", cat([plot(latent_models[n](latents[n])[1:10:end,1,1]|>c, width=2000) for n in 1:3]..., dims=1))
end;


# cat([plot(latent_models[n](latents[n])[1:10:(300*Int(fs_fender)),1,1]|>c) for n in 1:3]..., dims=1)

GC.gc(true); GC.gc(false)


function save(latents, amp_model)
    latents_cpu = latents|> c
    amp_model_cpu = amp_model |> c
    BSON.@save "../upsampling_model2.bson" latents_cpu amp_model_cpu
    @info "Saved: $(Dates.now())"
end

#=
begin
    training_done
    save(latents, amp_model)
    "Saved: $(Dates.now())"
end
=#
