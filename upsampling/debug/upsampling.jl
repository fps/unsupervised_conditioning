import Pkg
Pkg.activate(".")

import Flux, CUDA, cuDNN, DSP, Statistics, ProgressLogging, Images, BSON, Dates, Random, WAV, Format

c = Flux.cpu
g = Flux.gpu

@info "# Utility functions"

include("../plotting.jl")
include("../../basis.jl")

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

#= 
@info "# Upsampling lowpass filter design"

begin
    lo_pass = 0.25
    lo_stop = 0.5
    n_lo = filtord(lo_pass, lo_stop, 0.1, 0.00001, 1)
end;

@info "n_lo: $(n_lo)"

upsampling_coeff = DSP.Filters.digitalfilter(
    DSP.Filters.Lowpass(
        lo_pass
    ), 
    DSP.Filters.FIRWindow(
        DSP.Windows.hann(n_lo)
    ),
    fs=1
);

plot(upsampling_coeff)
=#

@info "# Composite latent model"

function latent_model(coeff_bp, k, coeff_lo)
    Flux.Chain(
        # Flux.Dropout(0.1, dims=2), 
        filtering_model(coeff_bp),
        upsampling_model(k, coeff_lo)
    )
end;

@info "# Reading the training data..."

import WAV

fs_fender = 48000

# T_start = 1# Int(8*fs_fender)
# T_end = Int(833*fs_fender)

begin
    output1, fs_output = WAV.wavread("../../data/fender_blues_junior/Take7_Audio 1-1.3.wav");  
    output = repeat(Float32.(output1), 1, 1, 1) |> g
    (size(output), fs_output, length(output)/fs_output)
end

begin
    input1, fs_input = WAV.wavread("../../data/fender_blues_junior/varying_noise_20_min_seed_0.wav"); 
    input = repeat(Float32.(input1), 1, 1, 1) |> g
    (size(input), fs_input, length(input)/fs_input)
end

begin
    @assert(fs_input == fs_output)
    fs_fender = fs_input
end

@info "# Control data..."

control_periods = [
    [2*length(output) / fs_fender  (1/10)*length(output) / fs_fender],
    [(1/20)*length(output) / fs_fender  (1/200)*length(output) / fs_fender],
    [5 0.2],
]

control_frequencies = [1.0 ./ x for x in control_periods]

maximum_downsamplings = [
    [maximum_downsampling(fs_fender, max(1/10, maximum(c))) for c in control_frequencies]...
]

ks = halfings.(maximum_downsamplings)

downsamplings = [ 1 / k^2 for k in ks ]

n_ctrls = length(ks)

@info "# Filter design..."

bandpass_filter_ords = [
    filtord(
        maximum(control_frequencies[n]),
        2*maximum(control_frequencies[n]), 
        0.01, 
        0.01,
        fs_fender/(2^ks[n])
    )

    for n in 1:n_ctrls
]

bandpass_filter_coeffs = [
    DSP.Filters.digitalfilter(
        DSP.Filters.Bandpass(control_frequencies[n]...), 
        DSP.Filters.FIRWindow(DSP.Windows.hann(bandpass_filter_ords[n])),
        fs = fs_fender / downsamplings[n]
    )
    for n in 1:n_ctrls
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

    for n in 1:n_ctrls
]

latent_models = [
    latent_model(bandpass_filter_coeffs[n], ks[n], lowpass_filter_coeffs)
    
    for n in 1:n_ctrls
] |> g;

latent_bandpass_models = [ latent_models[n][1] for n in 1:n_ctrls ]

# #=
@info "# Creating amp model and latent data"

@info "## Latents"

latents = [
    1.0f-10* randn(Float32, latent_sizes[n], 1, 1)

    for n in 1:n_ctrls
] |> g;


@info "Local variance filters"

bandpassed_latent_sizes = length.(latents) .- (bandpass_filter_ords .- 1)


@info "## Amp model"

# begin
#     amp_model_width=32
#     
#     amp_model = Flux.Chain(
#         Flux.Conv((1,), (1+n_ctrls)=>amp_model_width),
#         [Flux.Conv((3,), amp_model_width=>amp_model_width, Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
#         [Flux.Conv((3,), amp_model_width=>amp_model_width, Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
#         Flux.Conv((1,), amp_model_width=>1)
#     ) |> g
# end;

include("../AmpModels.jl")

# begin
# 	struct residual_model
# 		layers
#     function residual_model(layers)
#       new(layers)
#     end
# 
# 		function residual_model(n_channels::Int, amp_model_width::Int)
# 			new([
# 				Flux.Conv((1,), n_channels=>amp_model_width),
# 		        [ Flux.Conv((3,), (amp_model_width+n_channels)=>amp_model_width, Flux.tanh, dilation=d)	for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
# 		        [Flux.Conv((3,), (amp_model_width+n_channels)=>amp_model_width, Flux.tanh, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
# 		        Flux.Conv((1,), (amp_model_width+n_channels)=>1)
# 			])
# 		end
# 	end
# 	
# 	function(m::residual_model)(data)
# 		input = copy(data)
# 		N_input = size(input, 1)
# 		data = m.layers[1](data)
# 		for n in 2:length(m.layers)
# 			data = m.layers[n](cat(input[(1+N_input-size(data, 1)):end,:,:], data, dims=2))
# 		end
# 		data
# 	end
# end

#amp_model = AmpModels.residual_model(1+n_ctrls, 32) |> g
amp_model = AmpModels.model(1+n_ctrls, 32) |> g

# =#

#=
@info "# Loading previously saved model and latent data"

saved = BSON.load("../upsampling_model2.bson")

latents = saved[:latents_cpu] |> g

amp_model = saved[:amp_model_cpu] |> g
=#


offset = 48000 - size(amp_model(randn(Float32, 48000, 4, 1) |> g), 1)

@info "Local variance models"
# local_variance_filtords = [
#     filtord(
#         0.5*minimum(control_frequencies[n]),
#         minimum(control_frequencies[n]),
#         0.01,
#         0.001,
#         fs_fender / 2^ks[n]
#     )
#     for n in 1:n_ctrls
# ]

# local_variance_filter_coeffs = [
#     if (bandpassed_latent_sizes[n]) < local_variance_filtords[n]
#         ones(bandpassed_latent_sizes[n]) ./ bandpassed_latent_sizes[n]
#     else
#         DSP.digitalfilter(DSP.Lowpass(minimum(control_frequencies[n])), DSP.FIRWindow(DSP.hann(local_variance_filtords[n])))
#         # ones(local_variance_filtords[n]) ./ local_variance_filtords[n]
#     end
#     for n in 1:n_ctrls
# ]

local_variance_filtords = [
    min(bandpassed_latent_sizes[n], Int(ceil(
        (fs_fender / 2^ks[n]) * (1/minimum(control_frequencies[n]))
    )))

    for n in 1:n_ctrls
]

local_variance_filter_coeffs = [
    # ones(local_variance_filtords[n]) ./ local_variance_filtords[n]
    gaussian(local_variance_filtords[n]/2, local_variance_filtords[n]/2, 1:local_variance_filtords[n])


    for n in 1:n_ctrls
]

local_variance_filter_coeffs = [
    local_variance_filter_coeffs[n] ./ sum(local_variance_filter_coeffs[n])
    for n in 1:n_ctrls
]


local_variance_models = filtering_model.(local_variance_filter_coeffs) |> g


@info "# Data splitting...."

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

# opt_amp_model = Flux.setup(Flux.Optimisers.AdamW(0.01), amp_model)

# opt_latents = Flux.setup(Flux.Optimisers.AdamW(0.01), latents)
# opt_latents = Flux.setup(Flux.Optimisers.AdamW(0.0001, (0.9, 0.999), 0.001), latents)

opt = Flux.setup(Flux.AdamW(0.001), [ latents, amp_model ])
#opt = Flux.setup(Flux.AdamW(0.001, (0.9, 0.999), 0.001), [ latents, amp_model ])

function format(n, p=8)
    Format.format(n, precision=p)
end

function train()
    losses = []
    perm = Random.randperm(N_segments)
    # ProgressLogging.@progress for p in 1:N_segments
    for p in 1:N_segments
        n = perm[p] - 1

        amp_loss = 0
        variance_loss = 0
        mean_loss = 0

        loss, grad = Flux.withgradient([latents, amp_model]) do (latents, amp_model)
            local ls = data_latent(latents, n)
            
            local l = cat([latent_models[m](
                ls[m]) for m in 1:n_ctrls]..., dims=2)
            
            local x = cat(
                input[(1+n*2^ks[1]):((n+1)*2^ks[1]), :, :], 
                l[1:(2^ks[1]), :, :], 
                dims=2
            )

            local y = output[(offset+1+n*2^ks[1]):((n+1)*2^ks[1]), :, :]

            amp_loss = Statistics.mean((amp_model(x) .- y).^2)

            bp = [ latent_bandpass_models[n](latents[n]) for n in 1:n_ctrls ]

            variance_loss = 1f0 * sum(Statistics.mean.([ (1f0 .- local_variance_models[n](bp[n].^2)).^2 for n in 1:n_ctrls])) / N_segments
            mean_loss = 1f0 * sum(Statistics.mean.([ local_variance_models[n](bp[n]).^2 for n in 1:n_ctrls])) / N_segments

            # the_loss += 1f-4 * Statistics.mean(l.^2)
            
            amp_loss # + variance_loss + mean_loss
        end
        # global the_grad = grad
        # Flux.update!(opt_latents, latents, grad[1][1])
        # Flux.update!(opt_amp_model, amp_model, grad[1][2])
        Flux.update!(opt, [ latents, amp_model], grad[1])
        @info "loss: $(format(loss)), amp_loss: $(format(amp_loss)), mean_loss: $(format(mean_loss)), variance_loss: $(format(variance_loss)), step: $(p), segment: $(n)"
        push!(losses, loss)
    end
    @info "mean loss: $(Statistics.mean(losses))"
    the_latents = [ latent_models[n](latents[n]) for n in 1:n_ctrls ]
    limits = [ (minimum(the_latents[n]), maximum(the_latents[n])) for n in 1:n_ctrls ]
    @info "limits: $(limits)"
    Images.save("latents.png", cat([plot(the_latents[n][1:10:end,1,1]|>c, width=6000) for n in 1:3]..., dims=1))
end;


# cat([plot(latent_models[n](latents[n])[1:10:(300*Int(fs_fender)),1,1]|>c) for n in 1:3]..., dims=1)

GC.gc(true); GC.gc(false)


function save(filename, latent_models, latents, amp_model)
    latents_cpu = latents|> c
    amp_model_state_cpu = Flux.state(amp_model |> c)
    latent_models_state_cpu = Flux.state(latent_models |> c)
    BSON.@save filename latent_models_state_cpu latents_cpu amp_model_state_cpu
    @info "Saved: $(Dates.now())"
end

#=
begin
    training_done
    save(latents, amp_model)
    "Saved: $(Dates.now())"
end
=#
