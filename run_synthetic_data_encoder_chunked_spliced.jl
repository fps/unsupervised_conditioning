@info "Training..."

# epoch = 1

losses = []

batchsize = 2^5

sigreg_test_points = dev(Float32.(collect(range(-5, 5, 17))))
sigreg_N_slices = 16

train_amp = true

n_epochs = 3000
n_warmup = 50
n_encoder_only = 100

lr_encoder_max = 5f-5
lr_encoder_min = 1f-6

lr_amp_max = 5f-4
lr_amp_min = 1f-5

lr_encoder = Float32.(cat(range(lr_encoder_min, lr_encoder_max, n_warmup), range(lr_encoder_max, lr_encoder_min, n_epochs-n_warmup), dims=1))
lr_amp = Float32.(cat(zeros(n_encoder_only), range(lr_amp_min, lr_amp_max, n_warmup), lr_amp_max * ones(n_epochs-n_warmup-n_encoder_only), dims=1))

for epoch in 1:length(lr_amp)
    # global epoch
    global losses
    global losses_mse
    global train_amp

    data_loader = Flux.MLUtils.DataLoader(chunks_dev, batchsize=batchsize, shuffle=true, partial=false)

    epoch_losses = []

    Flux.adjust!(opt[1], lr_encoder[epoch])
    Flux.adjust!(opt[2], lr_amp[epoch])

    train_amp = epoch > n_encoder_only
    
    for (x1, y1, x2, y2, y1_spec, y2_spec, supervision1, supervision2) in ProgressBars.ProgressBar(data_loader)
        directions = dev(randn(Float32, sigreg_N_slices, 3))
        
        for n in 1:sigreg_N_slices
            directions[n,:] ./ LinearAlgebra.norm(directions[n,:])
        end

        fft_noises = [1f-1 .* randn(Float32, fft_size), 2] |> dev
        
        loss, grad = Flux.withgradient([m_encoder, m_amp]) do (m_encoder, m_amp)
            encoded1 = m_encoder(y1_spec)
            encoded2 = m_encoder(y2_spec)

            l_supervision = 0 
            l_similarity = 0
            l_mse = 0
            
            l_limits = Flux.mse(min.(1, max.(-1, encoded1)), encoded1) + Flux.mse(min.(1, max.(-1, encoded2)), encoded2)

            l_supervision += 1f1 * Statistics.mean([Flux.mse(supervision1[1:1,:] .* encoded1, supervision1[2:4, :]), Flux.mse(supervision2[1:1,:] .* encoded2, supervision2[2:4, :])])

            l_similarity += 1f-1 * Flux.mse(encoded1, encoded2)

            if train_amp
                mean_encoded = 5f-1 .* (encoded1 .+ encoded2)
                processed = m_amp(cat(x2, permutedims(repeat(mean_encoded, 1, 1, size(x2, 1)), (3, 1, 2)), dims=2)[:,:,:])
                l_mse = 1f0 * Flux.mse(y2[(1+m_amp_offset):end, :, :], processed)
            end

            (
                l_supervision + l_similarity + l_mse + l_limits, 
                l_mse,
                l_limits,
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
        titles = ["Total loss", "MSE", "Limits loss"]
        for n in 1:3
            UnicodePlots.lineplot(cpu(m_encoder(chunks_dev[5]))[n,:], width=:auto, ylim=[-1.1, 1.1], xlim=[1, size(chunks_dev[5], 2)]) |> display
        end
    
        # UnicodePlots.lineplot(log10.(losses), width=:auto) |> display
        for n in 1:length(losses[1])
            UnicodePlots.lineplot(log10.([losses[k][n] for k in 1:length(losses)]), width=:auto, title=titles[n]) |> display
        end
    end
    
    epoch += 1

    if epoch >= 10000
        break
    end
end
