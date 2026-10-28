import EmpiricalCDFs

function uniform_marginals(data)
    new_data = copy(data)
    
    for n in 1:size(data, 2)
        cdf = EmpiricalCDFs.EmpiricalCDF(data[:,n])
        sort!(cdf)
        new_data[:,n] = cdf.(data[:,n])
    end

    new_data
end

import Distances, OptimalTransport

function sinkhorn_from_uniform(data, reg)
    sources = rand(size(data)...)
    C = Distances.pairwise(Distances.SqEuclidean(), sources, data, dims=1)
    plan = OptimalTransport.sinkhorn(ones(size(data, 1)), ones(size(data, 1)), C, reg)
    targets = copy(sources)
    for n in 1:size(data, 1)
        targets[n,:] = sum(plan[n,:] .* data, dims=1)
    end

    (plan, sources, targets)
end
    
