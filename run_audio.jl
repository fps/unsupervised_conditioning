import Pkg
Pkg.activate(".")

import Plots
import Flux
import CUDA

import Statistics

g = Flux.gpu
c = Flux.cpu
g = g

freq = 50

t = Float32.(repeat(0:(1/1000):1, 1, 1, 1)) |> g
x = sin.(2*pi*freq .* t) |> g
y = tanh.(5 .* sin.(2*pi*3.1 .* t) .* sin.(2*pi*freq .* t)) .+ sin.(2*pi*10.8 .* t) |> g

t = 0:(1/1000):1 |> g

num_basis_functions = 30
# basis = Float32.(cat([[sin.(f .* t .+ phase) for t in t, f in 2*pi .* (1:num_basis_functions)] for phase in [0, pi/2]]..., dims=2))
# basis = cat(basis, ones(1001,1), (1/1000).*(0:(1/1000):1), dims=2)
# basis = cat(ones(1001,1), 0:(1/1000):1, dims=2)
basis = Float32.([exp.(-num_basis_functions^2 .* (t.-center).^2) for t in t, center in 0:(1/num_basis_functions):1]) |> g

latent_params = Float32(0.0001) * randn(Float32, size(basis, 2), 2) |> g

# latent = zeros(Float32, size(t)...)

width = 16

model = Flux.Chain(
    Flux.Conv((1,), 3=>width), 
    [Flux.Conv((1,), width=>width, Flux.rrelu) for n in 1:4]...,
    Flux.Conv((1,), width=>1)
) |> g

opt = Flux.setup(Flux.AdamW(), [model, latent_params])

loss(m, x, y) = Flux.Losses.mse(m(x), y)

for m in 1:500;
    losses = []
    for n in 1:50; 
        the_loss, the_grads = Flux.withgradient([model, latent_params]) do params
            latent = basis * params[2]
            loss(params[1], cat(x, latent, dims=2), y) + 0.1f0 * Statistics.mean(latent).^2.0f0 + 0.1f0 * (1.0f0 - Statistics.var(latent)).^2 + 0.1f0*(latent[:,1]' * latent[:,2])^2.0f0
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
        Plots.plot(y[:,1,1] |> c, title="Training output", legend=:none),
        Plots.plot(model(cat(x, basis * latent_params, dims=2))[:,1,1] |> c, title="Model output", legend=:none),
        Plots.plot((basis * latent_params)[:,1,1] |> c, title="Inferred control input1", legend=:none),
        Plots.plot((basis * latent_params)[:,2,1] |> c, title="Inferred control input2", legend=:none),
        layout=(5,1),
        ytickfontsize=4,
        xtickfontsize=4
    ) |> display
end
