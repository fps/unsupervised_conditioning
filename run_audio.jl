import Pkg
Pkg.activate(".")

println("Importing packages...")

import Plots
import Measures

import Flux
import CUDA

import Statistics
import LinearAlgebra

println("Setting up things...")

g = Flux.gpu
c = Flux.cpu
g = g

freq = 500

N_data = 20000;

# t = Float32.(repeat(0:(1/2000):1, 1, 1, 1)) 
t = 0:(1/(N_data-1)):1

# x = sin.(2*pi*freq .* t) 
x = randn(N_data, 1)

# y = tanh.(5 .* sin.(2*pi*3.1 .* t) .* sin.(2*pi*freq .* t)) .+ sin.(2*pi*10.8 .* t) 

control1 = 0.5f0 .* sin.(2*pi*3*t.+0.1) .+ 1.0f0
control2 = cos.(2*pi*13*t) 

# y = control1 .* tanh.(x .* 5.0f0 .* control2)

y = tanh.(control2 .+ control1 .* x)

y = Float32.(repeat(y, 1, 1, 1)) |> g
x = Float32.(repeat(x, 1, 1, 1)) |> g

num_basis_functions = 80
# basis = Float32.(cat([[sin.(f .* t .+ phase) for t in t, f in 2*pi .* (1:num_basis_functions)] for phase in [0, pi/2]]..., dims=2))
# basis = cat(basis, ones(1001,1), (1/1000).*(0:(1/1000):1), dims=2)
# basis = cat(ones(1001,1), 0:(1/1000):1, dims=2)
basis = Float32.([exp.(-num_basis_functions.^2 .* (1 .* (t.-center)).^2) for t in t, center in 0:(1/(num_basis_functions-1)):1]) |> g
basis = basis ./ sum(basis, dims=2)
    
# basis = basis./sum(basis, dims=2)
    
num_latent_variables = 2

I = zeros(num_latent_variables, num_latent_variables)
for n in 1:num_latent_variables
    I[n,n] = 1
end
    
latent_params = 0.00001f0 * randn(Float32, size(basis, 2), num_latent_variables) |> g

# latent = zeros(Float32, size(t)...)

println("Setting up model...")

width = 16

model = Flux.Chain(
    Flux.Conv((1,), (1+num_latent_variables)=>width), 
    [Flux.Conv((1,), width=>width, Flux.selu) for n in 1:8]...,
    Flux.Conv((1,), width=>1)
) |> g

opt = Flux.setup(Flux.AdamW(), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)

println("Entering training looop...")

for m in 1:5000;
    losses = []
    for n in 1:50; 
        
        print(".")
        the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
            latent = basis * params[2]
            loss(params[1], cat(x, latent, dims=2), y) # + 0.00001f0 * (latent[:,1]' * latent[:,2])^2.0f0 # + 0.00001f0 * Statistics.mean(latent).^2.0f0 # + 0.00000001 * sum((1.0f0 .- Statistics.var(latent, dims=2))).^2# + 0.00001 * sum(abs.(latent)) # #  + sum((I - Statistics.cov(latent_params)).^2)# + 10.0f0 * Statistics.mean(diff(latent, dims=1).^2)# 0.01f0 * Statistics.mean(latent.^2) #  + 1.0f0 * (1.0f0 - Statistics.var(latent)).^2 + Statistics.mean(latent.^2) + 0.1f0*(latent[:,1]' * latent[:,2])^2.0f0 # + 0.1f0 * (1.0f0 - Statistics.var(latent)).^2 + 0.1f0*(latent[:,1]' * latent[:,2])^2.0f0
            # + sum(([1 0] .- latent[1,:]).^2)
                
            #LinearAlgebra.norm(Statistics.cor(latent, dims=1) - [1 0; 0 1])
            #  + Statistics.mean(diff(latent, dims=1).^2) + Statistics.mean(latent).^2 + (1 - Statistics.var(latent)).^2
        end
        push!(losses, the_loss)
        
        Flux.update!(opt, [model, latent_params], the_grads[1])
    end; 
    display((m, Statistics.mean(losses)))
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        # Plots.plot(control1[:,:,1] |> c, title="Control 1", legend=:none),
        # Plots.plot(control2[:,:,1] |> c, title="Control 2", legend=:none),
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(cat(x, basis * latent_params, dims=2))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot((basis * latent_params)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm
    ) |> display
end
