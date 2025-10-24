import Flux, DSP

function downsampler(stages, taps, channels, cutoff=0.5)
    coeffs = Float32.(DSP.digitalfilter(DSP.Lowpass(cutoff), DSP.FIRWindow(DSP.Windows.hamming(taps))))

    coeffs2 = Float32.([0, 1, 0])

    Flux.Chain(
        [Flux.Chain(Flux.Conv(repeat(coeffs, 1, 1, channels), groups=channels), Flux.Conv(repeat(coeffs2, 1, 1, channels), stride=2, groups=channels)) for n in 1:stages]...
    )
end

function upsampler(stages, taps, channels, cutoff=0.5)
    coeffs = Float32.(DSP.digitalfilter(DSP.Lowpass(cutoff), DSP.FIRWindow(DSP.Windows.hamming(taps))))

    coeffs2 = Float32.([0, 2, 0])

    Flux.Chain(
        [Flux.Chain(Flux.ConvTranspose(repeat(coeffs2, 1, 1, channels), stride=2, groups=channels), Flux.Conv(repeat(coeffs, 1, 1, channels), groups=channels)) for n in 1:stages]...
    )
end
