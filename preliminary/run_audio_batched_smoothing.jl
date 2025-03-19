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


println("Initial setup...")

include("../basis.jl")
include("../varying_noise.jl")

g = Flux.gpu
c = Flux.cpu
g = g


println("Creating synthetic training data...")

FS = 48000.0f0

T = 1.0f0

blocksize = Int(FS/4)

t = Float32.(0:(1/FS):(T-(1/FS)))

N_data = length(t)

# x = randn(Float32, N_data, 1, 1) |> g
x = 10.0f0 .* varying_noise(N_data, 1000, 8, FS) |> g

f1 = 6.1653f0
f2 = 36.3421f0
p2 = 1/f2

control1 = 1.0f0 .* (sin.(2*pi*f1*t.+0.1) .+ 1.2f0)  |> g
# f1 = 5.0f-1
# control1 = collect(0:(1/N_data):(1-(1/N_data))) |> g

control2 = 1.0f0 .* cos.(2*pi*f2*t) |> g

y = tanh.(control1 .+ control2 .* x)

x = x ./ 10.0f0

sigmas = Float32.([1/f1, 1/f2] ./ 8)
num_latent_variables = length(sigmas)

lower_bandwidth = 2
upper_bandwidth = 2
band1 = (f1 / lower_bandwidth, f1 * upper_bandwidth)
band2 = (f2 / lower_bandwidth, f2 * upper_bandwidth)

downsampling = 20

upsampling_model = Flux.ConvTranspose([(i == j ? 1.0f0 : 0.0f0) for k = 1:downsampling, i = 1:num_latent_variables, j = 1:num_latent_variables], stride=downsampling) |> g

smoothing_kernel_half_length = Int(ceil(3 * maximum(sigmas) * FS/(2*downsampling)))

smoothing_kernels = ones(2*smoothing_kernel_half_length+1, 2)

smoothing_kernels[:,1] = DSP.Filters.digitalfilter(DSP.Filters.Bandpass((band1./(24000/downsampling))...), DSP.FIRWindow(smoothing_kernels[:,1]))

smoothing_kernels[:,2] = DSP.Filters.digitalfilter(DSP.Filters.Bandpass((band2./(24000/downsampling))...), DSP.FIRWindow(smoothing_kernels[:,2]))

smoothing_weights = zeros(2*smoothing_kernel_half_length+1, 2, 2)
for n in 1:num_latent_variables
    smoothing_weights[:,n:n,n:n] = smoothing_kernels[:,n:n,:]
end

smoothing_weights = smoothing_weights |> g
smoothing_model = Flux.Conv(smoothing_weights) |> g

reconstruction_kernel = DSP.Filters.digitalfilter(DSP.Filters.Lowpass((band2[2] / 24000)...), DSP.FIRWindow(ones(downsampling*2*smoothing_kernel_half_length+1)))

reconstruction_weights = zeros(downsampling*2*smoothing_kernel_half_length+1, 2, 2)
for n in 1:num_latent_variables
    reconstruction_weights[:,n:n,n:n] = reconstruction_kernel
end

reconstruction_model = Flux.Conv(reconstruction_weights) |> g

if init
    latent_params = 1.0f-5 * randn(Float32, Int(ceil((length(x)/downsampling)))+(4*smoothing_kernel_half_length), num_latent_variables) |> g
    # scales = 1.0f0 * randn(Float32, 1, 1, num_latent_variables, 1)  |> g
end


println("Dividing data into blocks...")

index_blocksize = Int(ceil((blocksize / downsampling)))

x_blocks = reshape(x, blocksize, 1, :) |> g
y_blocks = reshape(y, blocksize, 1, :) |> g
# latent_params_blocks = reshape(latent_params, blocksize, num_latent_variables, :) |> g
index_blocks = reshape(1:(Int(ceil(length(x) / downsampling))), index_blocksize, :) |> g


println("Setting up model...")

width = 8

if init 
    model = Flux.Chain(
        Flux.Conv((1,), (1+num_latent_variables)=>width), 
        [Flux.Conv((1,), width=>width, Flux.tanh) for n in 1:8]...,
        Flux.Conv((1,), width=>1)
    ) |> g

    opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])
end
        
loss(m, x, y) = Flux.Losses.mse(m(x), y)


println("Entering training loop...")

# time, value pairs per control
finetuning = [
    [
        [0, T],
        [-0.1f0, 0.1f0]
    ],
    [
        [[t for t in 0:p2:T];[t for t in (p2/2):p2:T]],
        [[-1.0f-1 for t in 0:p2:T];[1.0f-1 for t in (p2/2):p2:T]]
    ]
] |> g

# Profile.@profile 
for m in 1:20000;
    the_latent = reconstruction_model(upsampling_model(smoothing_model(reshape(latent_params, :, num_latent_variables, 1))))
    # the_latent = latent(gaussian, centers, sigmas, reshape(t, :, 1, 1, 1)|>g, latent_params) - latent(gaussian, centers, 10 .* sigmas, reshape(t, :, 1, 1, 1)|>g, latent_params)
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(cat(x, the_latent, dims=2))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot((the_latent)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.02,
    ) |> display

    # Plots.plot(latent_params |> c) |> display
    losses = []
    sample_losses = []
    for n in 1:1; 
        batchsize = 1
        data_loader = Flux.MLUtils.DataLoader((x_blocks, y_blocks, index_blocks), batchsize=batchsize, shuffle=true)
        print("<")
        
        local sample_loss = 0
        for (x_block, y_block, index_block) in data_loader
            # print((size(x_block), size(y_block), size(index_block)))
            # display(upsampling_model(smoothing_model(cat([latent_params[index_block[1, b]:(2*smoothing_kernel_half_length + index_block[end, b]), :] for b in 1:size(x, 3)]..., dims=3))))
            print(".")
            the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
                # l = loss(params[1], cat(x_block, [params[2][index_block, d, :] for d in 1:num_latent_variables]..., dims=2), y_block)
                
                local smoothed_latent = smoothing_model(cat([params[2][index_block[1, b]:(4*smoothing_kernel_half_length + index_block[end, b]), :] for b in 1:size(x, 3)]..., dims=3))
                
                local upsampled_latent = upsampling_model(smoothed_latent)
                
                local reconstructed_latent = reconstruction_model(upsampled_latent)
                
                # lp = cat([params[2][index_block[:,b], :] for b in 1:size(x_block, 3)]..., dims=3)
                l = loss(params[1], cat(x_block, reconstructed_latent, dims=2), y_block)
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
