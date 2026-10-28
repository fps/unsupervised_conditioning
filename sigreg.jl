module SIGReg

import LinearAlgebra
import Flux

struct SigReg
    evaluation_points
end

SigReg() = SigReg(collect(range(-5, 5, 17)))

function random_direction(D)
    p = randn(Float32, D)
    p ./= LinearAlgebra.norm(p)
    p
end

function sample_characteristic_function(samples, evaluation_points; dims=1)
    outer = evaluation_points .* samples
    (1/length(samples) .* sum(exp.(-im .* outer), dims=dims))
end

function normal_characteristic_function(evaluation_points)
    exp.(-evaluation_points.^2 ./ 2)
end

function sigreg(x, test_points, directions)
    l = 0

    N_slices = size(directions, 1)
    
    for n in 1:N_slices
        projected = directions[n:n,:] * x

        # @info "$(size(projected))"
        
        l += (1/N_slices) * Flux.mse(normal_characteristic_function(test_points), sample_characteristic_function(projected, test_points, dims=2))
    end

    l
end



end
