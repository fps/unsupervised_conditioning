include("varying_noise.jl")
import SVF

function synthetic_data_3controls(T=1, fs=48000)
    N_data = fs * T
    x = varying_noise(N_data, 1024, 400, 8, fs, Nothing)

    t = 0:(1/fs):(T-(1/fs))
    f = [1/(110.7), 33.3, 6.1]

    controls = [
        # 2 .+ sin.(2 * pi * f[1] .* t .+ 7.33), 
        0.5 .+ 0.5 .* (0:(1/N_data):(1-(1/N_data))),
        0.5 .+ 2f-1 .* sin.(2 * pi * f[2] .* t .+ pi/4),
        0.5 .+ 2f-1 .* sin.(2 * pi * f[3] .* t .+ -pi/8)
    ]

    controls[1] = reverse(controls[1])

    states = [SVF.State(), SVF.State(), SVF.State()]

    # distorted = tanh.(10 .* controls[1] .* x)
    distorted = tanh.(100 .* x)

    filtered1 = zeros(size(x))
    for n in 1:length(x)
        filtered1[n] = SVF.process(states[1], distorted[n], 1, 0, 0, 5000/fs, 0.707)
    end

    filtered2 = zeros(size(x))
    for n in 1:length(x)
        filtered2[n] = SVF.process(states[2], distorted[n], 0, 1, 0, 2000/fs, 0.707)
    end

    filtered3 = zeros(size(x))
    for n in 1:length(x)
        filtered3[n] = SVF.process(states[3], distorted[n], 0, 0, 1, 1000/fs, 0.707)
    end

    y = controls[1] .* filtered1 .+ controls[2] .* filtered2 .+ controls[3] .* filtered3

    # sigmas = [ 20, 1, 0.1 ]
    sigmas = (1 ./ f) ./ 5

    x, y, fs, sigmas, controls, t
end
