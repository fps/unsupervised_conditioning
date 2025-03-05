function gaussian(center, sigma, t)
    exp.(-(t.-center).^2 ./ sigma.^2)
end

function mexican_hat(center, sigma, t)
   (1 .- ((t .- center) ./ sigma).^2) .* exp.(-(t .- center).^2 ./ (2*sigma.^2))
end

function basis(f, centers, sigmas, t_blocks)
    b = f(centers, sigmas, t_blocks)
    # b ./ (sum(b, dims=2) .+ 0.00000001f0)
    # b
    b ./ sum(b, dims=1)
end


function latent(f, centers, sigmas, t_blocks, latent_params)
    selectdim(sum(latent_params .* basis(f, centers, sigmas, t_blocks), dims=2), 2, 1)
end

