@info "Setting up supervised controls..."

supervised_chunk_size = 12 * m_amp_offset

# supervised_controls = [[0.0, 0.0, 0.0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
supervised_control_points = map(x -> Float32.(x), ([[i, j, k] for i in [-1,0,1] for j in [-1, 0, 1] for k in [-1, 0, 1]]))
supervised_control_points = supervised_control_points[Random.randperm(length(supervised_control_points))]
extended_supervised_control_points = cat([0, 0, 0], supervised_control_points..., [0, 0, 0], dims=2)
# extended_supervised_control_points .*= 0.5
supervised_controls = cat([cat([cat(collect(range(extended_supervised_control_points[c, n-1], extended_supervised_control_points[c, n], 2 * supervised_chunk_size)), extended_supervised_control_points[c, n] .* ones(supervised_chunk_size), dims=1) for n in 2:size(extended_supervised_control_points, 2)]..., dims=1) for c in 1:3]..., dims=2)'

# supervised_controls = [ 2 .* rand(Float32, 3) .- 1 for n in 1:27 ]

@info "Done."
