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

FS = 48000.0f0

T = 1.0f0

blocksize = 2000

t = Float32.(0:(1/FS):(T-(1/FS)))

N_data = length(t)

# x = sin.(40.0f0*2.0f0*pi.*t) .* randn(N_data) |> g
x = randn(Float32, N_data, 1, 1) |> g

f1 = 1.3653f0
f2 = 56.3421f0
p2 = 1/f2

control1 = 1.0f0 .* (sin.(2*pi*f1*t.+0.1) .+ 1.2f0)  |> g
# f1 = 5.0f-1
# control1 = collect(0:(1/N_data):(1-(1/N_data))) |> g

# control2 = 1.0f0 .* cos.(2*pi*f2*t) |> g

y = tanh.(control1 .+ control2 .* x)

num_basis_functions = Int(ceil(8 * maximum([f1, f2]) * T))
centers = T .* (0:(1/(num_basis_functions-1)):1)
centers = Float32.(reshape(centers, 1, :, 1, 1)) |> g

sigmas = Float32.([1/(8*f1), 1/(8*f2)])
sigmas = reshape(final_sigmas, 1, 1, :, 1) |> g

num_latent_variables = length(sigmas)

if init
    latent_params = 1.0f-5 * randn(Float32, 1, num_basis_functions, num_latent_variables, 1) |> g
    # scales = 1.0f0 * randn(Float32, 1, 1, num_latent_variables, 1)  |> g
end


println("Dividing data into blocks...")

x_blocks = reshape(x, blocksize, 1, :) |> g
y_blocks = reshape(y, blocksize, 1, :) |> g
t_blocks = reshape(t, blocksize, 1, 1, :) |> g


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

sample_loss = 0

# Profile.@profile 
for m in 1:2000;
    the_latent = latent(gaussian, centers, sigmas, reshape(t, :, 1, 1, 1)|>g, latent_params) - latent(gaussian, centers, 10 .* sigmas, reshape(t, :, 1, 1, 1)|>g, latent_params)
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(repeat(cat(x, the_latent, dims=2), 1, 1, 1))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot((the_latent)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        # layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.25,
    ) |> display

    losses = []
    sample_losses = []
    for n in 1:1; 
        batchsize = 1
        data_loader = Flux.MLUtils.DataLoader((x_blocks, y_blocks, t_blocks), batchsize=batchsize, shuffle=true)
        print("<")
        
        sample_loss = 0
        for (x_block, y_block, t_block) in data_loader
            # print((size(x_block), size(y_block), size(basis_block)))
            batchsize = size(x_block, 2)
            print(".")
            the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
                p = params[2]
                lt = latent(gaussian, centers, sigmas, t_block, p) .- latent(gaussian, centers, 10 .* sigmas, t_block, p)
                # l = loss(params[1], cat(x_block, latent(gaussian, centers, sigmas, t_block, p + max(0, (1.0f0 - (m/200))) * g(randn(Float32, size(p)))), dims=2), y_block) 
                l = loss(params[1], cat(x_block, lt, dims=2), y_block)
                # lp_l = latent(gaussian, centers, sigmas .* 20, reshape(centers, :, 1, 1, 1), p)
                sample_loss = l
                # l += 1.0f2 * Statistics.mean(lp_l.^2)
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
