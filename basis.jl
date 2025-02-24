function gaussian(center, sigma, t)
    exp.(-(t.-center).^2 ./ sigma.^2)
end

function mexican_hat(center, sigma, t)
   (1 .- ((t .- center) ./ sigma).^2) .* exp.(-(t .- center).^2 ./ (2*sigma.^2))
end
