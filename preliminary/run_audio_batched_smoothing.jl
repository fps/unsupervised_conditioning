import Pkg
Pkg.activate(".")

println("Importing packages...")

import Plots
import Measures

import Flux
import CUDA

import Statistics
import LinearAlgebra

import Random
import Profile
import DSP

include("../divisors.jl")

if init
    println("Initial setup...")

    include("../varying_noise.jl")

    global g = Flux.gpu
    global c = Flux.cpu
    global g = g


    println("Creating synthetic training data...")

    global FS = 48000.0f0

    global T = 1.0f0

    global blocksize = Int(FS/8)

    global t = Float32.(0:(1/FS):(T-(1/FS)))

    global N_data = length(t)

    # x = randn(Float32, N_data, 1, 1) |> g
    global x = varying_noise(N_data, 1000, 8, FS) |> g

    function triangle(t)
        2 .* ifelse.(mod.(t, 2*pi) .< pi, mod.(t, pi)/pi , 1 .- mod.(t.-pi, pi)/pi) .- 1
    end

    global f1 = 2.1653f0
    global f2 = 81.3421f0

    global control1 = 1.0f0 .* (sin.(2*pi*f1*t.+0.1) .+ 1.2f0)  |> g
    # f1 = 5.0f-1
    # control1 = collect(0:(1/N_data):(1-(1/N_data))) |> g

    global control2 = (1.0f0 .* cos.(2*pi*f2*t)).^1 |> g

    global y = tanh.(control1 .+ control2 .* x * 30)

    global lower_bandwidth = 3
    global upper_bandwidth = 3

    # f1 = f2

    global bands = [
        f1 / lower_bandwidth  f1 * upper_bandwidth
        f2 / lower_bandwidth  f2 * upper_bandwidth
    ]

    global num_latent_variables = size(bands, 1)

    global downsampling = zeros(Int, num_latent_variables)

    for n in 1:num_latent_variables
        downsampling[n] = reverse(divisors(blocksize))[findmax(reverse((FS ./ (2*divisors(blocksize)) .> maximum(bands[n,:]))))[2]]
    end

    # global downsampling = [100 100]
    println("downsampling: $(downsampling)")

    global upsampling_models = [
        Flux.ConvTranspose(ones(Float32, d, 1, 1), stride=d) |> g
            
        for d in downsampling
    ]

    global smoothing_kernel_half_lengths = [
        Int(ceil((FS/downsampling[n]) / (2 * minimum(bands[n,:]))))
                                                            
        for n in 1:num_latent_variables
    ]

    global smoothing_kernel_lengths = smoothing_kernel_half_lengths .* 2 .+ 1

    # smoothing_window = ones
    global smoothing_window = DSP.Windows.hanning

    global smoothing_models = [
        Flux.Conv(
            repeat(
                DSP.Filters.digitalfilter(DSP.Filters.Bandpass(bands[n,:]...), DSP.FIRWindow(smoothing_window(smoothing_kernel_lengths[n])); fs=FS/downsampling[n])
                , 1, 1, 1
            )
        ) |> g

        for n in 1:num_latent_variables
    ]
    # reconstruction_window = ones
    global reconstruction_window = DSP.Windows.hanning

    global reconstruction_models = [
        Flux.Conv(
            repeat(
                DSP.Filters.digitalfilter(DSP.Filters.Lowpass(FS/(4*downsampling[n])), DSP.FIRWindow(reconstruction_window(downsampling[n]*2*smoothing_kernel_half_lengths[n]+1)); fs=FS)
                , 1, 1, 1
            )
        ) |> g
            
        for n in 1:num_latent_variables
    ]

    global latent_models = [
        x -> x |> smoothing_models[n] |> upsampling_models[n] |> reconstruction_models[n]
        for n in 1:num_latent_variables
    ]

    global latent_params = [
        1.0f-5 * randn(Float32, Int(ceil((length(x)/downsampling[n])))+(4*smoothing_kernel_half_lengths[n]), 1, 1) |> g
        
        for n in 1:num_latent_variables
    ]
    
    # scales = 1.0f0 * randn(Float32, 1, 1, num_latent_variables, 1)  |> g
    
    println("Dividing data into blocks...")

    global x_blocks = reshape(x, blocksize, 1, :) |> g
    global y_blocks = reshape(y, blocksize, 1, :) |> g
    # latent_params_blocks = reshape(latent_params, blocksize, num_latent_variables, :) |> g
    global index_blocks = reshape(1:length(x), blocksize, :) |> g


    println("Setting up model...")
    
    global width = 8
    global model = Flux.Chain(
        Flux.Conv((1,), (1+num_latent_variables)=>width), 
        [Flux.Conv((1,), width=>width, Flux.tanh) for n in 1:8]...,
        Flux.Conv((1,), width=>1)
    ) |> g

    global opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])
    
    global loss(m, x, y) = Flux.Losses.mse(m(x), y)
end
        


println("Entering training loop...")

# time, value pairs per control
# finetuning = [
#     [
#         [0, T],
#         [-0.1f0, 0.1f0]
#     ],
#     [
#         [[t for t in 0:p2:T];[t for t in (p2/2):p2:T]],
#         [[-1.0f-1 for t in 0:p2:T];[1.0f-1 for t in (p2/2):p2:T]]
#     ]
# ] |> g

function downsample_index(index, factor)
    Int(round(index / factor, RoundUp))
end

# Profile.@profile 
for m in 1:20000;
    global the_latent = [
        latent_params[n] |> latent_models[n]
        for n in 1:num_latent_variables
    ]
    
    # println(size.(the_latent))
    # the_latent = latent(gaussian, centers, sigmas, reshape(t, :, 1, 1, 1)|>g, latent_params) - latent(gaussian, centers, 10 .* sigmas, reshape(t, :, 1, 1, 1)|>g, latent_params)
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(cat(x, the_latent..., dims=2))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot(the_latent[n][:,1,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.02,
    ) |> display

    # sleep(0.1)
    
    # Plots.plot(latent_params |> c) |> display
    losses = []
    sample_losses = []
    @time for n in 1:1; 
        batchsize = 4
        data_loader = Flux.MLUtils.DataLoader((x_blocks, y_blocks, index_blocks), batchsize=batchsize, shuffle=true)
        print("<")
        
        local sample_loss = 0
        for (x_block, y_block, index_block) in data_loader
            # print((size(x_block), size(y_block), size(index_block)))
            # display(upsampling_model(smoothing_model(cat([latent_params[index_block[1, b]:(2*smoothing_kernel_half_length + index_block[end, b]), :] for b in 1:size(x, 3)]..., dims=3))))
            print(".")
            the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
                local the_latent_blocks = [
                    cat([
                        params[2][n][
                            downsample_index(index_block[1,b], downsampling[n]):(downsample_index(index_block[end, b], downsampling[n]) + 4*smoothing_kernel_half_lengths[n]),
                            :,
                            :
                        ]
                        for b in 1:size(x_block, 3)
                    ]..., dims=3)                    
                    for n in 1:num_latent_variables
                ]
                        
                # println("size.(the_latent_blocks): $(size.(the_latent_blocks))")
                
                local the_latent = cat([
                    reconstruction_models[n](upsampling_models[n](smoothing_models[n](
                        the_latent_blocks[n]
                    )))
                    for n in 1:num_latent_variables
                ]..., dims=2)
                
                # println("size(the_latent): $(size(the_latent))")
                
                # lp = cat([params[2][index_block[:,b], :] for b in 1:size(x_block, 3)]..., dims=3)
                l = loss(params[1], cat(x_block, the_latent, dims=2), y_block)
                # lp_l = latent(gaussian, centers, sigmas .* 20, reshape(centers, :, 1, 1, 1), p)

                sample_loss = l

                # l += sum(Statistics.mean(lp, dims=1).^2)
                # l += Statistics.mean(lp.^2)
                # l += sum(Statistics.mean(diff(lp, dims=1).^2, dims=1).^2)
                # l += (1 - Statistics.mean(diff(lp, dims=2).^2)).^2
                # l += sum((1.0f0 .- Statistics.mean(lp.^2, dims=1)).^2)
                # l += 1.0f0 * Statistics.mean((lp_l .- Statistics.mean(lp_l, dims=1)).^2)
                    # + 1.0f0 * sum(latent(gaussian, centers, sigmas .* 40, reshape(centers, :, 1, 1, 1), p).^2)
                
                # l_centers = latent(gaussian, centers, sigmas, reshape(centers, :, 1, 1, 1), p)
                # l += 1.0f0 * sum((1 .- Statistics.var(l_centers, dims=1)).^2)
                
                if finetune
                    for c in 1:num_latent_variables
                        l += sum((finetuning[c][2] .- latent(gaussian, centers, sigmas[:, :, c:c, :], reshape(finetuning[c][1], :, 1, 1, 1), p[:,:,c:c,:])).^2)
                    end
                end
                
                # if finetune
                #     l += 
                #     (
                #         sum((fine1[1] .- latent(gaussian, centers, sigmas, zeros(Float32, 1, 1, 1, 1) |> g, params[2])[:,1:1,:]).^2)
                #         + sum((fine1[2] .- latent(gaussian, centers, sigmas, T .* ones(Float32, 1, 1, 1, 1) |> g, params[2])[:,1:1,:]).^2)
                #         + sum((fine2[1] .- latent(gaussian, centers, sigmas, Float32.(reshape(collect(0:p2:T), :, 1, 1, 1)) |> g, params[2])[:,2:2,:]).^2)
                #         + sum((fine2[2] .- latent(gaussian, centers, sigmas, Float32.(reshape(collect((p2/2):p2:T), :, 1, 1, 1)) |> g, params[2])[:,2:2,:]).^2)
                #         + (max(maximum(fine1), maximum(latent(gaussian, centers, sigmas, Float32.(reshape(collect(0:0.01:T), :, 1, 1, 1)) |> g, params[2])[:,:,:])) - maximum(fine1)).^2
                #         + (min(minimum(fine1), minimum(latent(gaussian, centers, sigmas, Float32.(reshape(collect(0:0.01:T), :, 1, 1, 1)) |> g, params[2])[:,:,:])) + minimum(fine1)).^2
                #     )
                # end
                
                l
            end
            push!(losses, the_loss)
            push!(sample_losses, sample_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
        end
        print(">")
    end
    display((m, Statistics.mean(losses), Statistics.mean(sample_losses)))
end
