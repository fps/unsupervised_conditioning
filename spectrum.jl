import FFTW
import DSP

function spectrum(x, fft_size, fft_hop)
    N = size(x, 1)
    starts = 1:fft_hop:(N-(fft_size-1))
    w = DSP.Windows.hann(fft_size)
    Statistics.mean(cat([abs.(FFTW.fft(w .* x[start:(start+fft_size-1)]))[1:div(fft_size,2)] for start in starts]..., dims=2), dims=2)
end
