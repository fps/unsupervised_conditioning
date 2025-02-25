function cov(x)
    m = [Statistics.mean(x[:,i]) for i in 1:size(x,2)]

    [sum( (x[:,i] .- m[i]) .* (x[:,j] .- m[j])) for i in 1:size(x, 2), j in 1:size(x,2)] ./ (size(x,1) - 1)
end

# eye = zeros(Float32, num_latent_variables, num_latent_variables)
# for i in 1:num_latent_variables
#     eye[i,i] = 1.0f0
# end
# eye = g(eye)

function i2r(x)
    real.(x) .+ imag.(x)
end

function eye(num_latent_variables)
    I = zeros(num_latent_variables, num_latent_variables)
    for n in 1:length(num_latent_variables)
        I[n,n] = 1
    end
    I
end
