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

latent_params = 0.0001f0 * Float32.(randn(size(t, 1), 2)) |> g

num_latent_variables = size(latent_params, 2)

I = zeros(num_latent_variables, num_latent_variables)
for n in 1:num_latent_variables
    I[n,n] = 1
end
    

# latent = zeros(Float32, size(t)...)

println("Setting up model...")

width = 4

model = Flux.Chain(
    Flux.Conv((1,), (1+num_latent_variables)=>width), 
    [Flux.Conv((1,), width=>width, Flux.rrelu) for n in 1:16]...,
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
        
        print(".")
        the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
            loss(params[1], cat(x, params[2], dims=2), y) + 1000.0f0 * (sum(diff(params[2], dims=1).^2)/length(t)  +  sum(diff(diff(params[2], dims=1), dims=2).^2)/length(t) + sum(diff(diff(diff(params[2], dims=1), dims=2), dims=2).^2)/length(t) + sum(diff(diff(diff(diff(params[2], dims=1), dims=2), dims=2), dims=2).^2)/length(t))
        end
        push!(losses, the_loss)
        
        Flux.update!(opt, [model, latent_params], the_grads[1])
    end; 
    display((m, Statistics.mean(losses)))
    
    Plots.plot(
        Plots.plot(x[:,1,1] |> c, title="Training input", legend=:none), 
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(cat(x, latent_params, dims=2))[:,1,1] |> c, title="Model output", legend=:none),
        [ Plots.plot((latent_params)[:,n,1] |> c, title="Inferred control input $(n)", legend=:none) for n in 1:num_latent_variables]...,
        layout=(3+num_latent_variables,1),
        ytickfontsize=4,
        xtickfontsize=4,
        titlefontsize=4,
        margin=1Measures.mm,
        padding=1Measures.mm,
        linewidth=0.25
    ) |> display
end
