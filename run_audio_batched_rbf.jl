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


println("Initial setup...")

include("basis.jl")

g = Flux.gpu
c = Flux.cpu
g = g


println("Creating synthetic training data...")

FS = 48000

T = 2

blocksize = 1000

t = 0:(1/FS):(T-(1/FS))

t_blocks = reshape(t, 1000, :)

N_data = length(t)

x = sin.(40.0f0*2.0f0*pi.*t) .* randn(N_data) |> g

control1 = 0.5f0 .* sin.(2*pi*2.25*t.+0.1) .+ 1.0f0 |> g
control2 = cos.(2*pi*6.9*t) |> g

y = tanh.(control2 .+ control1 .* x) |> g

num_basis_functions = 40 * T

sigmas = [1/20, 1/5]
basis_functions = [mexican_hat, mexican_hat]

num_latent_variables = length(sigmas)

basis = Float32.([f(center, sigma, t) for (sigma, f) in zip(sigmas, basis_functions), t in t, center in T .* (0:(1/(num_basis_functions-1)):1)]) |> g
    
latent_params = 0.00000001f0 * randn(Float32, num_latent_variables, num_basis_functions) |> g


println("Dividing data into blocks...")

x_blocks = reshape(x, blocksize, :)
y_blocks = reshape(y, blocksize, :)
# basis_blocks = reshape(permutedims(basis,(1, 3, 2)), length(sigmas), blocksize, num_basis_functions, :)
basis_blocks = cat([basis[:,start:(start+blocksize-1),:] for start in 1:blocksize:96000]..., dims=4)
# basis_blocks = Float32.(


println("Setting up model...")

width = 16

model = Flux.Chain(
    Flux.Conv((1,), (1+num_latent_variables)=>width), 
    [Flux.Conv((1,), width=>width, Flux.celu) for n in 1:4]...,
    Flux.Conv((1,), width=>1)
) |> g

opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)


println("Entering training loop...")

for m in 1:5000;
    latent = hcat([(latent_params[n:n,:] * basis[n,:,:]')' for n in 1:size(latent_params, 1)]...)
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(repeat(cat(x, latent, dims=2), 1, 1, 1))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot((latent)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.25,
    ) |> display

    losses = []
    for n in 1:50; 
        batchsize = 5
        data_loader = Flux.MLUtils.DataLoader((x_blocks, y_blocks, basis_blocks), batchsize=batchsize, shuffle=true)
        print("<")
        
        for (x_block, y_block, basis_block) in data_loader
            # print((size(x_block), size(y_block), size(basis_block)))
            batchsize = size(x_block, 2)
            print(".")
            the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
                local m = params[1]
                local latent_params = params[2]
                # latent = hcat([(latent_params[n:n,:] * basis[n,:,:]')' for n in 1:size(latent_params, 1)]...)
                local latent = cat([cat([basis_block[sigma,:,:,batch] * latent_params[sigma,:] for sigma in 1:length(sigmas)]..., dims=2) for batch in 1:batchsize]..., dims=3)
                # print((size(latent), size(x_block), size(y_block)))
                loss(m, cat(reshape(x_block, blocksize, 1, batchsize), latent, dims=2), reshape(y_block, blocksize, 1, batchsize))
            end
            push!(losses, the_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
        end
        print(">")
    end; 
    display((m, Statistics.mean(losses)))
end
