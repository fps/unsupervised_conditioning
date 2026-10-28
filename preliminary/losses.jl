import Statistics

function minimum_distance_loss(x, y; dims=2)
    d = sum((x .- y).^2, dims=dims)
    max(0, max(Statistics.mean(minimum(d, dims=1)), Statistics.mean(minimum(d, dims=3))) - 1f-1)
end
