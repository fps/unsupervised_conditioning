function cat2_causal(x1, x2)
    N1 = size(x1, 1)
    N2 = size(x2, 1)
    if N1 < N2
        cat(x1, x2[(1+N2-N1):end, :, :], dims=2)
    else
        cat(x1[(1+N1-N2):end, :, :], x2, dims=2)
    end
end

function cat2_centered(x1, x2)
    N1 = size(x1, 1)
    N2 = size(x2, 1)
    if N1 < N2
        cat(x1, x2[(1+div(N2-N1, 2)):(div(N2-N1, 2)+N1), :, :], dims=2)
    else
        cat(x1[(1+div(N1-N2, 2)):(div(N1-N2, 2)+N2), :, :], x2, dims=2)
    end
end
