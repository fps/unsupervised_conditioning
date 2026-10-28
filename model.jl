@info "Setting up model..."

fft_size = 1024
fft_hop = div(fft_size, 2)

m_encoder = Flux.Chain(
    Flux.Dense(fft_size => 64),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    # Flux.BatchNorm(64),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    # Flux.BatchNorm(64),
    Flux.Dense(64 => 64, Flux.leakyrelu),
    # Flux.BatchNorm(64),
    Flux.Dense(64 => 32, Flux.leakyrelu),
    Flux.Dense(32 => 16, Flux.leakyrelu),
    Flux.Dense(16 => 16, Flux.leakyrelu),
    # Flux.Dense(16 => 8, Flux.tanh),
    Flux.Dense(16 => 3),
) |> dev

m_amp = Flux.Chain(
    Flux.Conv((1, ), 4 => 16),
    Flux.SkipConnection(Flux.Chain([Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:10 ]...), (mx, x) -> (mx .+ x[(size(x, 1) - size(mx, 1) + 1):end, :, :])),
    [Flux.Conv((3,), 16 => 16, Flux.leakyrelu, dilation=2^(k-1)) for k in 1:10 ]...,
    Flux.Conv((1,), 16 => 1)
) |> dev

m_amp_offset = 48000 - size(m_amp(dev(randn(Float32, 48000, 4, 1))), 1)

@info "m_amp_offset: $m_amp_offset"


@info "Setting up optimizer..."

opt = Flux.setup(Flux.Adam(2f-4), [m_encoder, m_amp])

@info "Done."

# Flux.adjust!(opt[1], 2f-5)

