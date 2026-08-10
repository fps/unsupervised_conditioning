import Pkg
Pkg.activate(".")

import Flux, BSON, UnicodePlots, Statistics

include("../jepa.jl/src/jepa.jl")

trained = BSON.load("fender_upsampling.bson")
latent2 = trained[:latent_cpu][:,:]'
latent_pairs = (latent2[:,1:(end-1)], latent2[:,2:end])

m_embedding = Flux.Chain(Flux.Dense(8 => 8), [Flux.Dense(8 => 8, Flux.leakyrelu) for n in 1:4]..., Flux.Dense(8 => 4, Flux.leakyrelu), Flux.Dense(4 => 4))
opt = Flux.setup(Flux.Adam(), m_embedding)

N_random_directions = 3

epoch = 0
while true;
    global epoch
    losses = [];
    data_loader = Flux.MLUtils.DataLoader(latent_pairs, batchsize=32, partial=false);
    for data in data_loader;
        Ds = [JEPA.random_direction(4) for n in 1:N_random_directions]
        evaluation_points = [5 .* randn(Float32, 15) for n in 1:N_random_directions]
        
        l, g = Flux.withgradient(m_embedding) do m_embedding;
            e1 = m_embedding(data[1]);
            e2 = m_embedding(data[2]);
            l = Flux.mse(e1, e2)

            es = cat(e1, e2, dims=2)

            for n in 1:N_random_directions
                projections = Ds[n]' * es
                l += 1f-0 * (1/N_random_directions) * Flux.mse(JEPA.normal_characteristic_function(evaluation_points[n]), JEPA.sample_characteristic_function(projections', evaluation_points[n])')
            end
            # + 1f-2 * Flux.mse([1, 1, 1, 1], Statistics.std(e1, dims=3)) + Flux.mse([0, 0, 0, 0], Statistics.mean(e1.^2, dims=3));

            
            l
        end;
        Flux.update!(opt, m_embedding, g[1]); 
        push!(losses, l);
    end;
    @info "epoch: $(epoch), loss: $(Statistics.mean(losses))";

    if epoch % 50 == 0
        for n in 1:4; UnicodePlots.lineplot(m_embedding(latent2)[n,1:200], width=:auto) |> display; end
    end
    
    epoch += 1
end 
