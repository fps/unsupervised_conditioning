import Flux, CUDA, cuDNN, WAV, DSP, FFTW, DelimitedFiles, Random, Statistics

@info "Reading data..."
inputs = Float32.(DelimitedFiles.readdlm("data/synthetic/AmpVTS/inputs.txt"))
outputs = Float32.(DelimitedFiles.readdlm("data/synthetic/AmpVTS/outputs.txt"))
params = Float32.(DelimitedFiles.readdlm("data/synthetic/AmpVTS/params.txt"))

@info "Preprocessing data..."
inputs = inputs[2001:end,:]
output = outputs[2001:end,:]

@info "Creating features..."
fft_inputs = abs.(cat([FFTW.fft(inputs[:,n]) for n in 1:size(inputs, 2)]..., dims=2))
fft_outputs = abs.(cat([FFTW.fft(output[:,n]) for n in 1:size(inputs, 2)]..., dims=2))

x = fft_outputs ./ (fft_inputs .+ 1)
x = x[1:4000, :]
x = log10.(x)

y = params

@info "Creating datasets..."
n_train = Int(round(0.95 * size(inputs, 2)))
p = Random.randperm(size(inputs, 2))

x_train = x[:,p[1:n_train]]
y_train = y[:,p[1:n_train]]

x_val = x[:,p[(n_train+1):end]]
y_val = y[:,p[(n_train+1):end]]

if false
    @info "Normalizing data..."
    @info "Calculating statistics..."
    x_mean = Statistics.mean(x_train)
    x_std = Statistics.std(x_train)

    y_mean = Statistics.mean(y_train)
    y_std = Statistics.std(y_train)

    @info "Applying statistics..."
    x_train .-= x_mean
    x_train ./= x_std

    y_train .-= y_mean
    y_train ./= y_std

    x_val .-= x_mean
    x_val ./= x_std

    y_val .-= y_mean
    y_val ./= y_std
end

@info "Setting up model and optimizer..."
m = Flux.Chain(Flux.Dense(size(x, 1) => 10), Flux.Dense(10 => 10, Flux.leakyrelu), Flux.Dense(10=>10, Flux.leakyrelu), Flux.Dense(10=>3))

opt = Flux.setup(Flux.AdamW(; eta=1f-4, lambda=0.0001, couple=false), m)

losses = []
losses_val = []

@info "Training..."
n = 0
while true
    global n
    l, g = Flux.withgradient(m) do m
        Flux.mse(m(x_train), y_train)
    end
    l_val = Flux.mse(m(x_val), y_val)

    push!(losses, l)
    push!(losses_val, l_val)

    if n % 1000 == 0
        UnicodePlots.lineplot(log10.(losses), width=:auto) |> display
        UnicodePlots.lineplot(log10.(losses_val), width=:auto) |> display
        @info "$(n): losses: train: $(l), val: $l_val"
    end
    
    Flux.update!(opt, m, g[1])
    n += 1
end

