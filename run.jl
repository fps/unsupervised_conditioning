import Plots
import Flux
import Statistics

t = Float32.(repeat(0:(1/1000):1, 1, 1, 1))
x = sin.(100 .* t)
y = tanh.(10 .* t .* sin.(100 .* t))
latent = zeros(Float32, size(t)...)

width = 8

model = Flux.Chain(
    Flux.Conv((1,), 2=>width), 
    [Flux.Conv((1,), width=>width, Flux.leakyrelu) for n in 1:4]..., 
    Flux.Conv((1,), width=>1)
)

opt = Flux.setup(Flux.Adam(), [model, latent])

loss(m, x, y) = Flux.Losses.mse(m(x), y)

for m in 1:100;
    display(m);
    for n in 1:100; 
        the_loss, the_grads = Flux.withgradient([model, latent]) do params
            # display(model2)
            loss(params[1], cat(x, params[2], dims=2), y) + Statistics.mean(diff(params[2], dims=1).^2) + Statistics.mean(params[2]).^2
        end
        
        Flux.update!(opt, [model, latent], the_grads[1])
        # Flux.train!(loss, model, [(x, y)], opt); 
    end; 
    
    Plots.plot(
        Plots.plot(model(cat(x, latent, dims=2))[:,1,1]),
        Plots.plot(latent[:,1,1]), layout=(2,1)
    ) |> display
end
