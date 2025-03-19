import DSP
import Random

function varying_noise(length, cutoff, order, fs, seed = Nothing)
    if (seed != Nothing)
        Random.seed!(seed)
    end
    f = DSP.Filters.digitalfilter(DSP.Filters.Lowpass(cutoff), DSP.Filters.Butterworth(order), fs=fs)
    x = 2 .* rand(length, 1) .- 1
    y = 2 .* rand(length, 1) .- 1
    y_f = DSP.filt(f, y)
    y_f = y_f ./ maximum(abs.(y_f))
    x .* y_f
end
    
