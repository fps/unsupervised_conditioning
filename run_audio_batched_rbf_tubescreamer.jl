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

y, fs_y = WAV.wavread("Take12_screamer-1.wav")
y = Float32.(y)

FS = fs_x

blocksize = Int(FS/10)

# We throw away between 1 and (blocksize-1) data away here.

N_data = Int(floor(length(y) / blocksize) * blocksize)
# N_data = 48000 * 8

T = N_data / FS

x = x[1:N_data] |> g
y = y[1:N_data] |> g

t = Float32.(0:(1/FS):(T-(1/FS)))

t_reduce = 10

t_blocks = reshape(t, blocksize, 1, 1, :) |> g
t_blocks_reduced = t_blocks[1:t_reduce:end, :, :, :]

println("Creating basis functions...")

basis_functions_per_second = 2
num_basis_functions = Int(floor(basis_functions_per_second * T))

sigmas = Float32.([120, 1]) |> g

basis_functions = [mexican_hat, mexican_hat]

num_latent_variables = length(sigmas)
sigmas = reshape(sigmas, 1, 1, num_latent_variables, 1)

# basis = Float32.([f(center, sigma, t) for (sigma, f) in zip(sigmas, basis_functions), t in t, center in T .* (0:(1/(num_basis_functions-1)):1)])
centers = Float32.(collect(T .* (0:(1/(num_basis_functions-1)):1))) |> g
centers = reshape(centers, 1, length(centers), 1, 1)

latent_params = Float32.(0.00000001f0 * randn(1, num_basis_functions, num_latent_variables, 1)) |> g

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

width = 32

model = Flux.Chain(
    Flux.Conv((500,), (1+num_latent_variables)=>width), 
    [Flux.Conv((10,), width=>width, Flux.celu) for n in 1:8]...,
    Flux.Conv((500,), width=>1)
) |> g

test_out = model(randn(Float32, blocksize, (1+num_latent_variables), 1) |> g)
offset = blocksize - size(test_out, 1)
println("Offset: $(offset)")

opt = Flux.setup(Flux.Adam(0.0001), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)


println("Entering training loop...")

I = eye(num_latent_variables)

w_resample = zeros(Float32, t_reduce, 2, 2)
w[:,1,1] .= 1
w[:,2,2] .= 1
resample_model = Flux.ConvTranspose(w, stride=t_reduce) |> g

# Profile.@profile 
for m in 1:5000;
    # latent = hcat([(latent_params[n:n,:] * basis[n,:,:]')' for n in 1:size(latent_params, 1)]...)
    
    # Plots.plot(
    #     Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
    #     Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
    #     Plots.plot(model(repeat(cat(x, latent, dims=2), 1, 1, 1))[:,1,1] |> c, title="Model output", legend=:none),
    #     [ Plots.plot((latent)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
    #     layout=(3+num_latent_variables,1),
    #     ytickfontsize=4,
    #     xtickfontsize=4,
    #     titlefontsize=4,
    #     margin=1Measures.mm,
    #     padding=1Measures.mm,
    #     linewidth=0.25,
    # ) |> display

    losses = []
    for n in 1:1; 
        batchsize = 32
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
                # local latent = reshape(sum(latent_params .* gaussian(centers, sigmas, t_block), dims=2), blocksize, num_latent_variables, size(t_block, 4))
                # local the_latents = zeros(Float32, size(x_block, 1), num_latent_variables, batchsize) |> g
                # print(size(latent))
                #print((minimum(latent, dims=(1,3)),maximum(latent, dims=(1,3))))

                local the_latents = latent(gaussian, centers, sigmas, t_block, latent_params)
                the_latents = resample_model(the_latents)
                # the_latents = the_latents .+ g(0.01f0 .* randn(Float32, size(the_latents)))
                
               loss(m, cat(x_block, the_latents, dims=2), y_block[(1+offset):end, :, :] |> g) + 0.001f0 * sum((I .- cov(latent_params[1,:,:,1])).^2)
            end
            push!(losses, the_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
        end
        print(">")
    end; 
    display((m, Statistics.mean(losses)))
    for k in 1:length(sigmas)
      Plots.plot(latent_params[1, :, k, 1] |> c)|> display
      Plots.plot(reshape(latent(gaussian, centers, sigmas[:,:,k:k,:], t_blocks[:,:,:,1:160], latent_params[:,:,k:k,:]), blocksize*160, 1, 1)[:,1,1]|>c) |> display
    end
    # Plots.plot(latent(gaussian, centers, sigmas, t_blocks[:,:,:,1:10])[:,:,1]|>c) |> display
end
