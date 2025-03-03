include("varying_noise.jl")
import SVF

function synthetic_data_3controls()
    fs = 48000
    T = 100
    N_data = fs * T
    x = varying_noise(N_data, 20, 8, fs, 0)

    t = 0:(1/fs):(T-(1/fs))
    f = [1/(110.7), 1/8.3, 1.097]
    controls = 0.3 .* [2 .+ sin.(2 * pi * f[1] .* t), 2 .+ sin.(2 * pi * f[2] .* t), 2 .+ cos.(2 * pi * f[3] .* t)]
    states = [SVF.State(), SVF.State()]
    distorted = tanh.(10 .* controls[1] .* x)
    filtered1 = zeros(size(x))
    for n in 1:length(x)
        filtered1[n] = SVF.process(states[1], distorted[n], controls[3][n], 0, 0, 0.02, 0.707)
    end
    filtered2 = zeros(size(x))
    for n in 1:length(x)
        filtered2[n] = SVF.process(states[2], distorted[n], 0, 0, controls[2][n], 0.02, 0.707)
    end
    y = filtered1 .+ filtered2
    x, y, 48000.0, controls
end
