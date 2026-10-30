@info "Training..."

# epoch = 1

losses = []

# similarity_weights = dev([1 3 9]')
similarity_weights = dev([1f0 1f0 1f0]')


# STAHP

batchsize = 2^4

sigreg_test_points = dev(Float32.(collect(range(-5, 5, 17))))
sigreg_N_slices = 16

train_amp = true

n_epochs = 2000
n_warmup = 20

lr_amp = Float32.(cat(range(1f-5, 1f-4, n_warmup), 1f-4 * ones(n_epochs), dims=1))
lr_encoder = Float32.(cat(range(1f-5, 1f-4, n_warmup), range(1f-4, 1f-5, n_epochs), dims=1))

for epoch in 1:length(lr_amp)
    # global epoch
    global losses
    global losses_mse

    data_loader = Flux.MLUtils.DataLoader(chunks_dev, batchsize=batchsize, shuffle=true, partial=false)

    epoch_losses = []

    Flux.adjust!(opt[1], lr_encoder[epoch])
    Flux.adjust!(opt[2], lr_amp[epoch])
    
    for (x1, y1, x2, y2, x1_spec, y1_spec, x2_spec, y2_spec, supervision1, supervision2) in ProgressBars.ProgressBar(data_loader)
        directions = dev(randn(Float32, sigreg_N_slices, 3))
        
        for n in 1:sigreg_N_slices
            directions[n,:] ./ LinearAlgebra.norm(directions[n,:])
        end

        fft_noises = [1f-1 .* randn(Float32, fft_size), 2] |> dev
        
        loss, grad = Flux.withgradient([m_encoder, m_amp]) do (m_encoder, m_amp)
            #=
            encoded1 = m_encoder(cat(x1_spec, y1_spec, dims=1) .+ fft_noises[1])
            encoded2 = m_encoder(cat(x2_spec, y2_spec, dims=1) .+ fft_noises[2])
            =#
            
            encoded1 = m_encoder(cat(x1_spec, y1_spec, dims=1))
            encoded2 = m_encoder(cat(x2_spec, y2_spec, dims=1))

            l_supervision = 0 
            l_similarity = 0
            l_mse = 0
            
            l_limits = Flux.mse(min.(1, max.(-1, encoded1)), encoded1) + Flux.mse(min.(1, max.(-1, encoded2)), encoded2)

            l_supervision += 1f1 * Statistics.mean([Flux.mse(supervision1[1:1,:] .* encoded1, supervision1[2:4, :]), Flux.mse(supervision2[1:1,:] .* encoded2, supervision2[2:4, :])])

            l_similarity += 1f1 * Flux.mse(similarity_weights .* encoded1, similarity_weights .* encoded2)
            # sim_unsupervised = (1 .- supervision1[1:1,:]) .* (1 .- supervision2[1:1,:])
            
            # l_similarity += 1f1 * Flux.mse(similarity_weights .* encoded1 .* sim_supervised, similarity_weights .* encoded2 .* sim_supervised)
            # l_similarity += 1f1 * Flux.mse(similarity_weights .* encoded1 .* sim_unsupervised, similarity_weights .* encoded2 .* sim_unsupervised)

            if train_amp
                mean_encoded = 5f-1 .* (encoded1 .+ encoded2)
                processed = m_amp(cat(x2, permutedims(repeat(mean_encoded, 1, 1, size(x2, 1)), (3, 1, 2)), dims=2)[:,:,:])
                l_mse = 1f0 * Flux.mse(y2[(1+m_amp_offset):end, :, :], processed)
            end
            
            # processed1 = m_amp(cat(x1, permutedims(repeat(encoded1, 1, 1, size(x1, 1)), (3, 1, 2)), dims=2)[:,:,:])
            # processed2 = m_amp(cat(x2, permutedims(repeat(encoded2, 1, 1, size(x2, 1)), (3, 1, 2)), dims=2)[:,:,:])
            
            # l_mse1 = Flux.mse(y1[(1+m_amp_offset):end, :, :], processed1)
            # l_mse2 = Flux.mse(y2[(1+m_amp_offset):end, :, :], processed2)
            # l_mse1 = Flux.mse(y1[(1+m_amp_offset):end, :, :] .* sim_supervised, processed1 .* sim_supervised)
            # l_mse2 = Flux.mse(y2[(1+m_amp_offset):end, :, :] .* sim_supervised, processed2 .* sim_supervised)
            
            # l_mse += Statistics.mean(l_mse2 + l_mse2)

            (
                l_supervision + l_similarity + l_mse + l_limits, # l_similarity #= + l_var =# #= + l_sigreg =# #= + l_encoded_mean =# + (l_mse1 + l_mse2)/2,
                # l_similarity,
                l_mse,
                # l_supervision,
                l_limits,
                # l_sigreg,
                # l_encoded_mean,
                # (l_mse1 + l_mse2) / 2
            )
        end
        
        Flux.update!(opt, [m_encoder, m_amp], grad[1])

        push!(epoch_losses, loss)
    end

    N_losses = length(epoch_losses[1])
    
    mean_losses = zeros(N_losses)
    for l in 1:length(epoch_losses[1])
        mean_losses[l] = Statistics.mean([epoch_losses[k][l] for k in 1:length(epoch_losses)])
    end
    
    push!(losses, mean_losses)
    
    if (epoch - 1) % 1 == 0
        for n in 1:3
            UnicodePlots.lineplot(cpu(m_encoder(cat(chunks_dev[5], chunks_dev[6], dims=1)))[n,:], width=:auto, ylim=[-1.1, 1.1], xlim=[1, size(chunks_dev[5], 2)]) |> display
        end
    
        # UnicodePlots.lineplot(log10.(losses), width=:auto) |> display
        for n in 1:length(losses[1])
            UnicodePlots.lineplot(log10.([losses[k][n] for k in 1:length(losses)]), width=:auto) |> display
        end
    end
    
    epoch += 1

    if epoch >= 10000
        break
    end
end
