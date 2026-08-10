n_params = 3
n_points = 1000

points = rand(n_points, n_params)

function kde(data, query, bandwidth)
    d = 0
    s2p = sqrt(2*pi)
    for n in 1:size(data, 1)
        d += (1/(bandwidth*s2p)) * exp(-LinearAlgebra.norm(data[n,:] - query)^2 / (2*bandwidth^2))
    end
    d / size(data, 1)
end

