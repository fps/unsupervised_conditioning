function dict(activation)
    if activation == identity
        return ""
    end
    
    if activation == Flux.tanh
        return "tanh"
    end

    if activation == Flux.relu
        return "relu"
    end

    if activation == Flux.leakyrelu
        return "leaky_relu"
    end

    throw("Unhandled activation type")
end

function dict(l::Flux.Conv)
    @assert length(size(l.weight)) == 3
    d = Dict()

    d["type"] = "conv1d"
    d["dilation"] = l.dilation
    d["groups"] = l.groups
    d["stride"] = l.stride
    d["activation"] = dict(l.σ)
    d["kernel_size"] = [size(l.weight, 1)]
    d["weights"] = [ permutedims(l.weight, (3, 2, 1)), l.bias ]
    d["shape"] = [ nothing, nothing, size(l.weight, 3) ]

    d
end

function dict(c::Flux.Chain)
    d = Dict()

    d["layers"] = [ dict(l) for l in c ]

    d
end

