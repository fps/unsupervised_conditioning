import Flux, DSP, UnicodePlots, Statistics, Random

latent_lowpass_half_ord = 3
latent_lowpass_ord = latent_lowpass_half_ord * 2 + 1
latent_lowpass_coeff = DSP.digitalfilter(DSP.Lowpass(0.5), DSP.FIRWindow(DSP.hanning(latent_lowpass_ord))) 

x = randn(Float32, 1000) .+ sin.((1:1000) .* (2 * pi) ./ 1000) .+ sin.(5 .* (1:1000) .* (2 * pi) ./ 1000)

opt = Flux.setup(Flux.Adam(), x)

shifts = cat(-latent_lowpass_half_ord:-1, 1:latent_lowpass_half_ord, dims=1)
current_shifts_perm = 1:length(shifts)

epoch = 1
while true
    global epoch
    global current_shifts_perm

    if epoch % length(shifts) == 1
        current_shifts_perm = Random.randperm(length(shifts))
    end
    
    shift = shifts[current_shifts_perm[(epoch % length(shifts)) + 1]]
    # @info "shift: $shift"
    
    x_std = Statistics.std(x)
    
    l, g = Flux.withgradient(x) do x
        coeff1 = latent_lowpass_coeff[latent_lowpass_half_ord + 1]
        coeff2 = latent_lowpass_coeff[latent_lowpass_half_ord + 1 + shift]

        s = coeff1 + coeff2
        # s = abs(coeff1) + abs(coeff2)

        coeff1 /= s
        coeff2 /= s
        
        1f-0 * Flux.mse(x, coeff1.* x .+ coeff2 .* circshift(x, shift)) 
    end

    if epoch % 100 == 1
        @info "epoch: $epoch, loss: $l"
        UnicodePlots.lineplot(x, width=:auto) |> display
    end
    
    Flux.update!(opt, x, g[1])

    x .*= x_std / Statistics.std(x)
    x .-= Statistics.mean(x)

    epoch += 1
end
