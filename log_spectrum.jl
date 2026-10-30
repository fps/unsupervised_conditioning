module LogSpectrum

struct LogSpectrum
    freqs
    test_functions
    window
end

function LogSpectrum(window, fs, f_start, f_end, f_N)
    len = length(window)
    freqs = logrange(f_start, f_end, f_N)
    test_functions = [exp.(-im .* ((1:len) ./ fs) .* (2 * pi) .* f) for f in freqs]
    LogSpectrum(freqs, test_functions, window)
end

function(ls::LogSpectrum)(x)
    cat([ ls.test_functions[n]' * (x .* ls.window) for n in 1:length(ls.test_functions) ]..., dims=1)
end

end
