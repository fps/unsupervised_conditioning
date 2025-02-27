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

println("Setting up things...")

g = Flux.gpu
c = Flux.cpu
g = g

freq = 500

FS = 48000

T = 2

t = 0:(1/FS):(T-(1/FS))

N_data = length(t)

# t = Float32.(repeat(0:(1/2000):1, 1, 1, 1)) 
# t = 0:(1/(N_data-1)):1

# x = sin.(2*pi*freq .* t) 
x = randn(N_data) |> g

# y = tanh.(5 .* sin.(2*pi*3.1 .* t) .* sin.(2*pi*freq .* t)) .+ sin.(2*pi*10.8 .* t) 

# control1 = 0.5f0 .* sin.(2*pi*1.1*t.+0.1) .+ 1.0f0 |> g
control1 = cat(0:(2/N_data):(1-(1/N_data)), 1:-(2/N_data):(1/N_data), dims=1) |> g
control2 = cos.(2*pi*13*t) |> g

# y = control1 .* tanh.(x .* 5.0f0 .* control2)

y = tanh.(control2 .+ control1 .* x) |> g

# y = Float32.(repeat(y, 1, 1, 1)) |> g
# x = Float32.(repeat(x, 1, 1, 1)) |> g


num_basis_functions = 80
# basis = Float32.(cat([[sin.(f .* t .+ phase) for t in t, f in 2*pi .* (1:num_basis_functions)] for phase in [0, pi/2]]..., dims=2))
# basis = cat(basis, ones(1001,1), (1/1000).*(0:(1/1000):1), dims=2)
# basis = cat(ones(1001,1), 0:(1/1000):1, dims=2)
# basis = Float32.([exp.(-num_basis_functions.^2 .* (1 .* (t.-center)).^2) for t in t, center in 0:(1/(num_basis_functions-1)):1]) |> g
# basis = basis ./ sum(basis, dims=2)

# sigmas = [1/30, 1/10]
# basis = Float32.([exp.(-(t.-center).^2 ./ sigma^2) for t in t, center in 0:(1/(num_basis_functions-1)):1, sigma in sigmas]) |> g
# for n in 1:size(basis, 3)
#     basis[:,:,n] = basis[:,:,n] ./ sum(basis[:,:,n], dims=2)
# end
# latent_params = 0.00000001f0 * randn(Float32, size(basis, 2), num_latent_variables) |> g


num_latent_variables = 2
basis = [cos(2*pi*f*t/2) + im*sin(2*pi*f*t/2) for f in 0:29, t in t] |> g
latent_params = 0.00001f0 * (randn(Float32, num_latent_variables, size(basis, 1)) + im*randn(Float32, num_latent_variables, size(basis, 1))) |> g

# basis = basis./sum(basis, dims=2)
    

I = zeros(num_latent_variables, num_latent_variables)
for n in 1:num_latent_variables
    I[n,n] = 1
end
    

# latent = zeros(Float32, size(t)...)

println("Setting up model...")

width = 16

model = Flux.Chain(
    Flux.Conv((1,), (1+num_latent_variables)=>width), 
    [Flux.Conv((1,), width=>width, Flux.rrelu) for n in 1:8]...,
    Flux.Conv((1,), width=>1)
) |> g

opt = Flux.setup(Flux.AdamW(0.001), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)

println("Entering training loop...")

function cov(x)
    m = [Statistics.mean(x[:,i]) for i in 1:size(x,2)]

    [sum( (x[:,i] .- m[i]) .* (x[:,j] .- m[j])) for i in 1:size(x, 2), j in 1:size(x,2)] ./ (size(x,1) - 1)
end

eye = zeros(Float32, num_latent_variables, num_latent_variables)
for i in 1:num_latent_variables
    eye[i,i] = 1.0f0
end
eye = g(eye)

function i2r(x)
    real.(x) .+ imag.(x)
end

for m in 1:5000;
    losses = []
    for n in 1:50; 
        data_loader = Flux.MLUtils.DataLoader((basis, x, y), batchsize=4000, shuffle=true)
        print(":")
        
        for (basis, x, y) in data_loader
            print(".")
            the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
                # latent = basis * params[2]
                # latent = hcat([basis[:,:,n] * params[2][:,n] for n in 1:size(latent_params, 2)]...)
                latent = i2r.(params[2] * basis)'
                # print(size(latent))
                loss(params[1], repeat(cat(x, latent, dims=2), 1, 1, 1), y) + 1f2 * (sum(i2r(params[2][:,1]).^2) + sum(i2r.(params[2][1, 2:10]).^2) + sum(i2r.(params[2][2, 8:end]).^2))# + 0.001f0 * sum((eye .- g(cov(latent))).^2) # + 0.00001f0 * (latent[:,1]' * latent[:,2])^2.0f0 # + 0.00001f0 * Statistics.mean(latent).^2.0f0 # + 0.00000001 * sum((1.0f0 .- Statistics.var(latent, dims=2))).^2# + 0.00001 * sum(abs.(latent)) # #  # + 10.0f0 * Statistics.mean(diff(latent, dims=1).^2)# 0.01f0 * Statistics.mean(latent.^2) #  + 1.0f0 * (1.0f0 - Statistics.var(latent)).^2 + Statistics.mean(latent.^2) + 0.1f0*(latent[:,1]' * latent[:,2])^2.0f0 # + 0.1f0 * (1.0f0 - Statistics.var(latent)).^2 + 0.1f0*(latent[:,1]' * latent[:,2])^2.0f0
                # + sum(([1 0] .- latent[1,:]).^2)
                    
                #LinearAlgebra.norm(Statistics.cor(latent, dims=1) - [1 0; 0 1])
                #  + Statistics.mean(diff(latent, dims=1).^2) + Statistics.mean(latent).^2 + (1 - Statistics.var(latent)).^2
            end
            push!(losses, the_loss)
            
            Flux.update!(opt, [model, latent_params], the_grads[1])
        end
    end; 
    display((m, Statistics.mean(losses)))
    
    # latent = hcat([basis[:,:,n] * latent_params[:,n] for n in 1:size(latent_params, 2)]...)
    latent = i2r.(latent_params * basis)'

    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        # Plots.plot(control1[:,:,1] |> c, title="Control 1", legend=:none),
        # Plots.plot(control2[:,:,1] |> c, title="Control 2", legend=:none),
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(repeat(cat(x, latent, dims=2), 1, 1, 1))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot((latent)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.25
    ) |> display
end
