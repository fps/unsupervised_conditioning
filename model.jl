@info "Setting up model..."

m_encoder = Flux.Chain(
    Flux.Dense(real_spectrum_length(fft_size) => 128, Flux.leakyrelu),
    Flux.Dense(128 => 64, Flux.leakyrelu),
    Flux.Dense(64 => 32, Flux.leakyrelu),
    Flux.Dense(32 => 16, Flux.leakyrelu),
    Flux.Dense(16 => 16, Flux.leakyrelu),
    # Flux.Dense(16 => 8, Flux.leakyrelu),
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

lr = 1f-4

opt = Flux.setup(Flux.Adam(lr), [m_encoder, m_amp])
# opt = Flux.setup(Flux.AdamW(;eta=lr, lambda=0.1), [m_encoder, m_amp])

@info "Done."

# Flux.adjust!(opt[1], 2f-5)

