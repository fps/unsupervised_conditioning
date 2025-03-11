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
include("synthetic_data.jl")
include("data/fender_blues_junior/data.jl")

g = Flux.gpu
c = Flux.cpu
g = g


println("Loading training data...")

if false
    # xx, fs_x = WAV.wavread("noise-3.wav")
    # x, fs_x = WAV.wavread("Take5_Audio 2-1.wav")
    x, fs_x = WAV.wavread("varying_noise.wav")
    
    x = Float32.(x)
    
    y, fs_y = WAV.wavread("Take15_screamer-1.wav")
    y = Float32.(y)
    
    FS = fs_x
end

# x, y, FS, sigmas, controls = synthetic_data_3controls()
x, y, FS, sigmas, controls = data()
    

blocksize = Int(FS/2)

# We throw away between 1 and (blocksize-1) data away here.

N_data = Int(floor(length(y) / blocksize) * blocksize)
# N_data = 48000 * 8

T = N_data / FS

x = x[1:N_data] |> g
y = y[1:N_data] |> g

# t = Float32.(0:(1/FS):(T-(1/FS))
t = collect((0:(N_data - 1)) ./ FS)

t_reduce = Int(FS/2)
# t_reduce = 600

t_blocks = reshape(t, blocksize, 1, 1, :) |> g
t_blocks_reduced = t_blocks[1:t_reduce:end, :, :, :]

println("Creating basis functions...")

basis_functions_per_second = 2/minimum(sigmas)
num_basis_functions = Int(floor(basis_functions_per_second * T))

# sigmas = Float32.([20, 1, 0.1]) |> g
# sigmas = 0.01 .* sigmas
sigmas = sigmas |> g

basis_functions = [gaussian, mexican_hat, mexican_hat]

num_latent = length(sigmas)
sigmas = reshape(sigmas, 1, 1, num_latent, 1)

centers = Float32.(collect(T .* (0:(1/(num_basis_functions-1)):1))) |> g
centers = reshape(centers, 1, length(centers), 1, 1)

if init
  # latent_params = Float32.(0.00000000001f0 * randn(1, num_basis_functions, num_latent, 1)) |> g
  latent_params = Float32.(0.1f0 * randn(1, num_basis_functions, num_latent, 1)) |> g
  # latent_params = latent_params ./ maximum(abs.(latent_params), dims=2)
end

# function basis(f, centers, sigmas, t_blocks)
#     b = f(centers, sigmas, t_blocks)
#     b ./ (sum(b, dims=2) .+ 0.00000001f0)
#     # b
# end
# 
# 
# function latent(f, centers, sigmas, t_blocks, latent_params)
#     reshape(sum(latent_params .* basis(f, centers, sigmas, t_blocks), dims=2), size(t_blocks,1), size(sigmas, 3), size(t_blocks, 4)) 
# end


println("Dividing data into blocks...")

x_blocks = reshape(x, blocksize, 1, :)
y_blocks = reshape(y, blocksize, 1, :)

  
if init
  println("Setting up model...")
  
  width = [32, 16]
  
  activation = Flux.celu
  # activation = Flux.tanh
  model = Flux.Chain(
    Flux.Conv((1,), (1+num_latent)=>width[1]),
    [Flux.Conv((3,), width[1]=>width[1], activation, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
    Flux.Conv((3,), width[1]=>width[2], activation),
    [Flux.Conv((3,), width[2]=>width[2], activation, dilation=d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]...,
    Flux.Conv((1,), width[2]=>1)
  ) |> g
  
  test_out = model(randn(Float32, blocksize, (1+num_latent), 1) |> g)
  offset = blocksize - size(test_out, 1)
  println("Offset: $(offset)")
end
  

if init
  println("Setting up optimizer...")
  global lr = 0.001
  # opt = Flux.setup(Flux.Adam(lr), [model, latent_params])
  opt = Flux.setup(Flux.AdamW(lr), [model, latent_params])
end

loss(m, x, y) = Flux.Losses.mse(m(x), y)

println("Setting up resample model...")
w_resample = zeros(Float32, t_reduce, num_latent, num_latent)
for n in 1:num_latent
    w_resample[:,n,n] .= 1
end
resample_model = Flux.ConvTranspose(w_resample, stride=t_reduce) |> g


println("Entering training loop...")

# Profile.@profile 
if init
  adjusted = 0
end

I = eye(num_latent) 

for m in 1:5000;
    losses = []
    for n in 1:1; 
        batchsize = 8
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
                local the_latents = latent(gaussian, centers, sigmas, t_block, latent_params)
                the_latents = resample_model(the_latents)
                # the_latents = the_latents .+ g(0.01f0 .* randn(Float32, size(the_latents)))
                # local the_latents = zeros(Float32, size(x_block, 1), 3, size(x_block, 3)) |> g                

               loss(m, cat(x_block, the_latents, dims=2), y_block[(1+offset):end, :, :] |> g) + 1.0f0 * sum(latent(gaussian, centers, sigmas .* 40, reshape(centers, :, 1, 1, 1), latent_params).^2) # + sum((0.001 .- Statistics.var(latent_params, dims=2)).^2) # +  sum(latent_params.^2) # sum((g(0.001 .* ones(1, 3, 1)) .- sum((latent(gaussian, centers, sigmas, reshape(centers, :, 1, 1, 1), latent_params) .- latent(gaussian, centers, sigmas .* 80, reshape(centers, :, 1, 1, 1), latent_params)).^2 ./ length(centers), dims=1)).^2)
                # + 0.01f0 * sum((I .- cov(latent_params[1,:,:,1])).^2) + 0.0001f0 * sum(Statistics.mean(latent_params, dims=1).^2)
            end
            push!(losses, the_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
            # global latent_params = latent_params ./ maximum(abs.(latent_params), dims=2)
        end
        print(">")
    end; 
    display((lr, m, Statistics.mean(losses)))

    if Statistics.mean(losses) < 1f-3 && adjusted == 0
      global lr *= 0.5
      Flux.adjust!(opt, lr)
      global adjusted = 1
    end

    if Statistics.mean(losses) < 5f-4 && adjusted == 1
      global lr *= 0.5
      Flux.adjust!(opt, lr)
      global adjusted = 2
    end

    for k in 1:length(sigmas)
      UnicodePlots.lineplot(latent(gaussian, centers, sigmas, t_blocks_reduced[1:1, :, :, :], latent_params)[1,k,:]|>c, width=displaysize(stdout)[2]-20) |> display
    end
end
