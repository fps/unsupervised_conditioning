import DSP
import Random

function varying_noise(length=48000, swingin=1024, cutoff=10, order=8, fs=48000, seed = Nothing)
    if (seed != Nothing)
        Random.seed!(seed)
    end
    f = DSP.Filters.digitalfilter(DSP.Filters.Lowpass(cutoff), DSP.Filters.Butterworth(order), fs=fs)
    x = 2 .* rand(length+swingin, 1) .- 1
    y = 2 .* rand(length+swingin, 1) .- 1
    y_f = DSP.filt(f, y)
    y_f = y_f ./ maximum(abs.(y_f))
    (x .* y_f)[(swingin+1):end]
end
    
