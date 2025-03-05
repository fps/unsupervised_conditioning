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


println("Initial setup...")

include("../basis.jl")

g = Flux.gpu
c = Flux.cpu
g = g


println("Creating synthetic training data...")

FS = 48000

T = 1

blocksize = 2000

t = 0:(1/FS):(T-(1/FS))

t_blocks = reshape(t, blocksize, :) |> g

N_data = length(t)

# x = sin.(40.0f0*2.0f0*pi.*t) .* randn(N_data) |> g
x = randn(N_data, 1)|> g

f1 = 0.2653
f2 = 56.3421

# control1 = 1.0f0 .* (sin.(2*pi*f1*t.+0.1) .+ 1.2f0) 
f1 = 2
control1 = collect(0:(1/N_data):(1-(1/N_data)))


control2 = 1.0f0 .* (cos.(2*pi*f2*t) .+ 1.1f0) 

# control1 = control1 + 0.5f0 .* randn(Float32, size(control1)...) |> g
# control2 = control2 + 0.5f0 .* randn(Float32, size(control1)...) |> g

control1 = control1 |> g
control2 = control2 |> g

y = tanh.(control1 .+ control2 .* x) |> g

num_basis_functions = Int(ceil(4 * maximum([f1, f2]) * T))

sigmas = [1/(6*f1), 1/(4*f2)]
basis_functions = [gaussian, gaussian]

num_latent_variables = length(sigmas)

basis = Float32.([f(center, sigma, t) for (sigma, f) in zip(sigmas, basis_functions), t in t, center in T .* (0:(1/(num_basis_functions-1)):1)]) |> g
    
basis = basis ./ sum(basis, dims=3)
    
latent_params = 0.00000001f0 * randn(Float32, num_latent_variables, num_basis_functions) |> g


println("Dividing data into blocks...")

x_blocks = reshape(x, blocksize, :)
y_blocks = reshape(y, blocksize, :)
# basis_blocks = reshape(permutedims(basis,(1, 3, 2)), length(sigmas), blocksize, num_basis_functions, :)
basis_blocks = cat([basis[:,start:(start+blocksize-1),:] for start in 1:blocksize:N_data]..., dims=4)
# basis_blocks = Float32.(



println("Setting up model...")

width = 8

model = Flux.Chain(
    Flux.Conv((1,), (1+num_latent_variables)=>width), 
    [Flux.Conv((1,), width=>width, Flux.tanh) for n in 1:4]...,
    Flux.Conv((1,), width=>1)
) |> g

opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)


println("Entering training loop...")

# Profile.@profile 
for m in 1:200;
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
    for n in 1:10; 
        batchsize = 1
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
                loss(m, cat(reshape(x_block, blocksize, 1, batchsize), latent, dims=2), reshape(y_block, blocksize, 1, batchsize)) # + (1 - Statistics.var(latent_params[2,:])).^2 # + sum(latent_params.^2)/(num_latent_variables*num_basis_functions)
            end
            push!(losses, the_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
        end
        print(">")
    end
    display((m, Statistics.mean(losses)))
end
