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

    global f1 = 12.1653f0
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
    
    global latent_model = x -> cat([latent_models[n](x[n]) for n in 1:num_latent_variables]..., dims=2)

    global latent_params = [
        1.0f-5 * randn(Float32, Int(ceil((length(x)/downsampling[n])))+(4*smoothing_kernel_half_lengths[n]), 1, 1) |> g
        
        for n in 1:num_latent_variables
    ]
    

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

    # global opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])
    global opt_model = Flux.setup(Flux.AdamW(0.001), model)
    global opt_latent = Flux.setup(Flux.AdamW(0.001), latent_params)
    
    global loss(m, x, y) = Flux.Losses.mse(m(x), y)
end
        


println("Entering training loop...")

function downsample_index(index, factor)
    Int(round(index / factor, RoundUp))
end

# Profile.@profile 
for m in 1:20000;
    global the_latent = latent_model(latent_params)
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(cat(x, the_latent, dims=2))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot(the_latent[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.02,
    ) |> display

    losses = []
    sample_losses = []
    batchsize = 2
    data_loader = Flux.MLUtils.DataLoader((x_blocks, y_blocks, index_blocks), batchsize=batchsize, shuffle=true)
    print("<")
    
    local sample_loss = 0
    
    # latent_gradient = [
    #     zeros(Float32, size(latent_params[n])) |> g
    #     for n in 1:num_latent_variables
    # ]
    
    latent_gradient = zeros(Float32, size(the_latent)) |> g
    
    for (x_block, y_block, index_block) in data_loader
        print(".")
        the_loss, the_grads = Flux.withgradient([model, the_latent]) do params
            # local the_latent_blocks = [
            #     cat([
            #         params[2][n][
            #             downsample_index(index_block[1,b], downsampling[n]):(downsample_index(index_block[end, b], downsampling[n]) + 4*smoothing_kernel_half_lengths[n]),
            #             :,
            #             :
            #         ]
            #         for b in 1:size(x_block, 3)
            #     ]..., dims=3)                    
            #     for n in 1:num_latent_variables
            # ]
            #         
            # local the_latent = cat([
            #     reconstruction_models[n](upsampling_models[n](smoothing_models[n](
            #         the_latent_blocks[n]
            #     )))
            #     for n in 1:num_latent_variables
            # ]..., dims=2)
            
            local the_latent_block = cat([params[2][index_block[1,b]:index_block[end,b], :, :] for b in 1:size(x_block,3)]..., dims=3)
            
            l = loss(params[1], cat(x_block, the_latent_block, dims=2), y_block)

            sample_loss = l
            
            l
        end
        push!(losses, the_loss)
        push!(sample_losses, sample_loss)
        
        # Flux.update!(opt, [model, latent_params], the_grads[1])
        Flux.update!(opt_model, model, the_grads[1][1])
        latent_gradient .+= the_grads[1][2]
        # for n in 1:num_latent_variables
        #     latent_gradient[n] = latent_gradient[n] .+ the_grads[1][2][n]
        # end
    end
        
    if m % 5 == 0
        the_loss, the_grads = Flux.withgradient(latent_params) do params
            loss(latent_model, params, the_latent - latent_gradient)
        end
        Flux.update!(opt_latent, latent_params, the_grads[1])
    end
    print(">")

    display((m, Statistics.mean(losses), Statistics.mean(sample_losses)))
end
