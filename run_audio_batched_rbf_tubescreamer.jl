import Pkg
Pkg.activate(".")

println("Importing packages...")

import Plots
Plots.unicodeplots()
import Measures

import Flux
import CUDA

import Statistics
import LinearAlgebra

import Random
import Profile

import WAV


println("Initial setup...")

include("basis.jl")
include("util.jl")

g = Flux.gpu
c = Flux.cpu
g = g


println("Loading training data...")

# xx, fs_x = WAV.wavread("noise-3.wav")
# x, fs_x = WAV.wavread("Take5_Audio 2-1.wav")
x, fs_x = WAV.wavread("varying_noise.wav")

x = Float32.(x)

y, fs_y = WAV.wavread("Take15_screamer-1.wav")
y = Float32.(y)

FS = fs_x

blocksize = Int(FS/2)

# We throw away between 1 and (blocksize-1) data away here.

N_data = Int(floor(length(y) / blocksize) * blocksize)
# N_data = 48000 * 8

T = N_data / FS

x = x[1:N_data] |> g
y = y[1:N_data] |> g

t = Float32.(0:(1/FS):(T-(1/FS)))

t_reduce = 100

t_blocks = reshape(t, blocksize, 1, 1, :) |> g
t_blocks_reduced = t_blocks[1:t_reduce:end, :, :, :]

println("Creating basis functions...")

basis_functions_per_second = 2
num_basis_functions = Int(floor(basis_functions_per_second * T))

sigmas = Float32.([60, 3]) |> g

basis_functions = [mexican_hat, mexican_hat]

num_latent = length(sigmas)
sigmas = reshape(sigmas, 1, 1, num_latent, 1)

# basis = Float32.([f(center, sigma, t) for (sigma, f) in zip(sigmas, basis_functions), t in t, center in T .* (0:(1/(num_basis_functions-1)):1)])
centers = Float32.(collect(T .* (0:(1/(num_basis_functions-1)):1))) |> g
centers = reshape(centers, 1, length(centers), 1, 1)

latent_params = Float32.(0.00000001f0 * randn(1, num_basis_functions, num_latent, 1)) |> g

function basis(f, centers, sigmas, t_blocks)
    b = f(centers, sigmas, t_blocks)
    b ./ (sum(b, dims=2) .+ 0.00000001f0)
end


function latent(f, centers, sigmas, t_blocks, latent_params)
    reshape(sum(latent_params .* basis(f, centers, sigmas, t_blocks), dims=2), size(t_blocks,1), size(sigmas, 3), size(t_blocks, 4))    
end


println("Dividing data into blocks...")

x_blocks = reshape(x, blocksize, 1, :)
y_blocks = reshape(y, blocksize, 1, :)
# basis_blocks = reshape(permutedims(basis,(1, 3, 2)), length(sigmas), blocksize, num_basis_functions, :)
# basis_blocks = cat([basis[:,start:(start+blocksize-1),:] for start in 1:blocksize:N_data]..., dims=4)
# basis_blocks = Float32.(


println("Setting up model...")

width = [8, 16]

# model = Flux.Chain(
#     Flux.Conv((100,), (1+num_latent)=>width, Flux.celu), 
#     [Flux.Conv((100,), width=>width, Flux.celu) for n in 1:4]...,
#     Flux.Conv((100,), width=>1)
# ) |> g
model = Flux.Chain(
  Flux.Conv((3,), (1+num_latent)=>width[1]),
  [Flux.Conv((3,), width[1]=>width[1], Flux.celu, dilation=d) for d in [2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
  Flux.Conv((3,), width[1]=>width[2], Flux.celu),
  [Flux.Conv((3,), width[2]=>width[2], Flux.celu, dilation=d) for d in [2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
  Flux.Conv((3,), width[2]=>1)
) |> g

test_out = model(randn(Float32, blocksize, (1+num_latent), 1) |> g)
offset = blocksize - size(test_out, 1)
println("Offset: $(offset)")

opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)


println("Entering training loop...")

I = eye(num_latent)

w_resample = zeros(Float32, t_reduce, num_latent, num_latent)
w_resample[:,1,1] .= 1
w_resample[:,2,2] .= 1
resample_model = Flux.ConvTranspose(w_resample, stride=t_reduce) |> g

# Profile.@profile 
for m in 1:5000;
    losses = []
    for n in 1:1; 
        batchsize = 16
        data_loader = Flux.MLUtils.DataLoader((x_blocks, y_blocks, t_blocks_reduced), batchsize=batchsize, shuffle=true)
        print("<")
        
        for (x_block, y_block, t_block) in data_loader
            # print((size(x_block), size(y_block), size(basis_block)))
            # print(size(x_block))
            batchsize = size(x_block, 3)
            print("-")
            the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
                local m = params[1]
                local latent_params = params[2]
                # local latent = reshape(sum(latent_params .* gaussian(centers, sigmas, t_block), dims=2), blocksize, num_latent, size(t_block, 4))
                # local the_latents = zeros(Float32, size(x_block, 1), num_latent, batchsize) |> g
                # print(size(latent))
                #print((minimum(latent, dims=(1,3)),maximum(latent, dims=(1,3))))

                local the_latents = latent(gaussian, centers, sigmas, t_block, latent_params)
                the_latents = resample_model(the_latents)
                # the_latents = the_latents .+ g(0.01f0 .* randn(Float32, size(the_latents)))
                
               loss(m, cat(x_block, the_latents, dims=2), y_block[(1+offset):end, :, :] |> g) # + 0.01f0 * sum((I .- cov(latent_params[1,:,:,1])).^2) + 0.0001f0 * sum(Statistics.mean(latent_params, dims=1).^2)
            end
            push!(losses, the_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
        end
        print(">")
    end; 
    display((m, Statistics.mean(losses)))
    for k in 1:length(sigmas)
      UnicodePlots.lineplot(latent(gaussian, centers, sigmas, t_blocks_reduced, latent_params)[1,k,:]|>c, width=100) |> display
      # UnicodePlots.lineplot(latent(gaussian, centers, sigmas, t_blocks_reduced, latent_params)[1,2,:]|>c, width=100) |> display
      # Plots.plot(latent_params[1, :, k, 1] |> c)|> display
      # Plots.plot(reshape(latent(gaussian, centers, sigmas[:,:,k:k,:], t_blocks[:,:,:,1:160], latent_params[:,:,k:k,:]), blocksize*160, 1, 1)[:,1,1]|>c) |> display
    end
    # Plots.plot(latent(gaussian, centers, sigmas, t_blocks[:,:,:,1:10])[:,:,1]|>c) |> display
end
