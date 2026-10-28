function average_spectrum(model, nsamples, latent, windowsize, window_overlap)
    Statistics.mean(
        DSP.spectrogram(
            Flux.cpu(
                m_amp(
                    dev(
                        cat(
                            randn(Float32, nsamples, 1, 1),
                            repeat(
                                latent[(10*48000):(10*48000), :, :], nsamples, 1, 1),
                            dims=2))))[:,1,1],
            windowsize,
            window_overlap).power,
        dims=2)
end

function average_average_spectrum(model, nsamples, latent, windowsize, window_overlap, ntrials)
    Statistics.mean(
        cat(
            [average_spectrum(model, nsamples, latent, windowsize, window_overlap) for trial in 1:ntrials]),
            dims=2)
end
